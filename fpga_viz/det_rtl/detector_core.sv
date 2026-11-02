// fpga_viz copy of fpga_det/rtl/detector_core.sv: identical arithmetic and cycle count; adds
// observation-only tap outputs (tap_*, pool_tap_*, att_tap_*) for the demo telemetry block.
// Box-detector core: shared scalar MAC engine for the 11-layer INT8 backbone plus the
// soft-argmax box head. Rounding, saturation, pooling, softmax table and divisions match
// fpga_det.quantize.IntegerDetector exactly. CHW INT8 activations, INT32 bias/accumulation.
module detector_core #(
    parameter WEIGHTS_FILE="fpga_det/export/weights.mem",
    parameter PARAMETER_FILE="fpga_det/export/parameters.mem",
    parameter EXP_FILE="fpga_det/export/exp.mem"
)(
    input wire clk, rst, start,
    input wire load_we,
    input wire [15:0] load_addr,
    input wire [7:0] load_data,
    output reg busy=0, done=0,
    output wire [63:0] result_class_logits,   // 4 x INT16 Q8.8, class 0 in bits [15:0]
    output reg signed [15:0] result_presence=0, // INT16 Q8.8 presence logit
    output reg [15:0] result_cx=0, result_cy=0,  // box centre, Q0.16 of the 128-pixel input side
    output wire [31:0] result_size,              // 2 x INT16 Q8.8 width/height logits, width in [15:0]
    output reg [1:0] result_class=0,             // argmax of the class logits
    output reg [31:0] inference_cycles=0,
    output reg layer_done=0,
    output reg [3:0] completed_layer=0,
    // fpga_viz: observation-only taps (no effect on the computation or its cycle count).
    output reg tap_we=0,                 // one pulse per backbone activation written
    output wire [7:0] tap_data,
    output reg [3:0] tap_layer=0,
    output reg [6:0] tap_channel=0,
    output reg [9:0] tap_global=0,       // cfg_poff + out_channel, 0..703
    output reg [5:0] tap_y=0,tap_x=0,
    output reg pool_tap_we=0,
    output reg [6:0] pool_tap_channel=0,
    output reg [7:0] pool_tap_value=0,
    output reg att_tap_we=0,             // softmax weight of each 16x16 heat-map cell
    output reg [7:0] att_tap_index=0,
    output reg [15:0] att_tap_value=0
);
    `include "layers.svh"
    (* rom_style="block" *) reg signed [7:0] weights [0:32767];
    (* rom_style="block" *) reg [63:0] parameters [0:1023];
    (* rom_style="block" *) reg [15:0] exp_table [0:512];
    initial begin
        $readmemh(WEIGHTS_FILE,weights);
        $readmemh(PARAMETER_FILE,parameters);
        $readmemh(EXP_FILE,exp_table);
    end

    reg source_bank=0;
    reg [15:0] read_addr=0,write_addr=0;
    reg [14:0] weight_addr=0;
    reg [9:0] parameter_addr=0;
    reg [9:0] exp_addr=0;
    reg write_en=0;
    reg signed [7:0] write_data=0,weight_q=0;
    reg [63:0] parameter_q=0;
    reg [15:0] exp_q=0;
    always @(posedge clk) begin
        weight_q<=weights[weight_addr];
        parameter_q<=parameters[parameter_addr];
        exp_q<=exp_table[exp_addr];
    end

    // Ping-pong activation banks, 64 KiB each, Vivado simple-dual-port template: one write
    // process with a single address/data/enable, one separate registered read. Bank A's write
    // port is shared by image loading (while idle) and inference. Every layer reads one bank and
    // writes the other; per-layer offsets keep the head's tap map (bank B @0) intact.
    wire loading=load_we && !busy;
    reg write_bank=0; // source bank of the layer that produced write_data
    wire bank_a_we=loading || (write_en && write_bank);
    wire bank_b_we=!loading && write_en && !write_bank;
    wire [15:0] bank_a_waddr=loading ? load_addr:write_addr;
    wire [7:0] bank_a_wdata=loading ? load_data:write_data;
    (* ram_style="block" *) reg signed [7:0] activation_a [0:65535];
    (* ram_style="block" *) reg signed [7:0] activation_b [0:65535];
    reg signed [7:0] bank_a_q=0,bank_b_q=0;
    always @(posedge clk) if (bank_a_we) activation_a[bank_a_waddr]<=bank_a_wdata;
    always @(posedge clk) bank_a_q<=activation_a[read_addr];
    always @(posedge clk) if (bank_b_we) activation_b[write_addr]<=write_data;
    always @(posedge clk) bank_b_q<=activation_b[read_addr];
    reg read_bank=0;
    always @(posedge clk) read_bank<=source_bank;
    wire signed [7:0] source_q=read_bank ? bank_b_q:bank_a_q;

    localparam IDLE=0,LAYER=1,P_ADDR=2,P_WAIT=3,P_LOAD=4,BEGIN_PIXEL=5,
        C_ADDR=6,C_WAIT=7,C_MAC=8,C_NEXT=9,C_REQUANT=10,C_WRITE=11,C_ADVANCE=12,
        G_INIT=13,G_ADDR=14,G_WAIT=15,G_ACC=16,G_STORE=17,
        R_START=18,R_DONE=19,HC_START=20,HC_DONE=21,HF_START=22,HF_DONE=23,
        X_INIT=24,X_ADDR=25,X_WAIT=26,X_ACC=27,
        A_CH=28,A_ADDR=29,A_WAIT=30,A_ACC=31,A_DONE=32,CX_DONE=33,CY_DONE=34,
        SA_START=35,SA_DONE=36,ARGMAX=37,FINISH=38,
        D_P_ADDR=40,D_P_WAIT=41,D_P_LOAD=42,D_ADDR=43,D_WAIT=44,D_MAC=45,D_NEXT=46,D_RQ=47,D_STORE=48,
        D_LAST=49,V_RUN=50,X_MUL=51,A_MUL=52,A_DIVSET=53,C_DRAIN=54,D_MUL=55;
    reg [5:0] state=IDLE;
    reg [3:0] layer=0;
    reg [6:0] out_channel=0,in_channel=0;
    reg [5:0] out_y=0,out_x=0;
    reg [1:0] kernel_y=0,kernel_x=0;
    reg tap_valid=0;
    reg signed [31:0] bias=0,accumulator=0;
    reg [23:0] requant_multiplier=0;
    reg [7:0] requant_shift=0;
    (* use_dsp="yes" *) reg signed [63:0] requant_product=0;
    // MAC pipeline: the 64 KiB bank outputs are slow, so operands are registered before multiplying.
    // C_NEXT edge: operands; next edge: product; next edge: accumulate.
    reg signed [7:0] op_a=0,op_w=0;
    reg op_valid=0,op_tap=0;
    wire signed [15:0] mac_product=$signed(op_a)*$signed(op_w);

    // Layer dimensions are copied from the generated table into registers once per layer.
    reg [1:0] cfg_stride=1,cfg_kernel=1;
    reg cfg_pad=0,cfg_dw=0;
    reg [7:0] cfg_in_hw=0;
    reg [6:0] cfg_out_hw=0;
    reg [14:0] cfg_in_plane=0;
    reg [12:0] cfg_out_plane=0;
    reg [14:0] cfg_woff=0;
    reg [6:0] cfg_wstep=0,cfg_cin=0,cfg_cout=0;
    reg [3:0] cfg_karea=0;
    reg [9:0] cfg_poff=0;
    reg [15:0] cfg_src_off=0,cfg_dst_off=0;

    // Two-stage tap address pipeline (C_ADDR: partial products, C_WAIT: sums).
    reg signed [9:0] tap_iy,tap_ix;
    reg [6:0] tap_ic;
    always_comb begin
        tap_iy=$signed({3'b0,(cfg_stride==2) ? {out_y,1'b0}:{1'b0,out_y}})+$signed({8'b0,kernel_y})-$signed({9'b0,cfg_pad});
        tap_ix=$signed({3'b0,(cfg_stride==2) ? {out_x,1'b0}:{1'b0,out_x}})+$signed({8'b0,kernel_x})-$signed({9'b0,cfg_pad});
        tap_ic=cfg_dw ? out_channel:in_channel;
    end
    reg [21:0] a_chan=0;
    reg signed [18:0] a_row=0;
    reg signed [9:0] a_col=0;
    reg [13:0] w_out=0;
    reg [9:0] w_in=0;
    reg [3:0] w_tap=0;
    reg [19:0] o_chan=0;
    reg [12:0] o_row=0;
    reg signed [15:0] mac_term_d=0;
    reg mac_valid=0;
    reg rq_valid=0,layer_done_stage=0;

    // Requantisation (verified bit-exact form of fpga.quantize.requant, see classifier_core.sv):
    // r = (p + (p<0 ? h-1 : h)) >>> s, h = s ? 2^(s-1) : 0, then saturation.
    function automatic signed [63:0] round_bias(input negative,input [7:0] shift);
        reg signed [63:0] half;
        begin
            half=(shift==0) ? 64'sd0 : (64'sd1<<(shift-1));
            round_bias=(shift==0) ? 64'sd0 : (negative ? half-64'sd1 : half);
        end
    endfunction
    function automatic [7:0] saturate8(input signed [63:0] biased,input [7:0] shift);
        reg [63:0] upper;
        reg signed [63:0] shifted;
        begin
            upper=64'hffff_ffff_ffff_ffff<<({1'b0,shift}+9'd7);
            shifted=biased>>>shift;
            if (biased[63]) saturate8=8'd0;                       // ReLU: r <= 0
            else if (|(biased & upper)) saturate8=8'd127;          // r >= 128
            else saturate8=shifted[7:0];
        end
    endfunction
    function automatic [15:0] saturate16(input signed [63:0] biased,input [7:0] shift);
        reg [63:0] upper;
        reg signed [63:0] shifted;
        begin
            upper=64'hffff_ffff_ffff_ffff<<({1'b0,shift}+9'd15);
            shifted=biased>>>shift;
            if (!biased[63] && |(biased & upper)) saturate16=16'h7fff;          // r >= 32768
            else if (biased[63] && !(&(biased | ~upper))) saturate16=16'h8000;  // r < -32768
            else saturate16=shifted[15:0];
        end
    endfunction
    reg signed [63:0] rq_biased=0;
    reg [15:0] rq_addr=0;
    reg [3:0] rq_layer=0;
    reg [6:0] rq_channel=0;
    reg [9:0] rq_global=0;
    reg [5:0] rq_y=0,rq_x=0;
    assign tap_data=write_data;
    reg rq_bank=0,layer_done_pending=0;

    // ---------------------------------------------------------------- head storage
    reg signed [7:0] pooled [0:95];
    reg [7:0] att [0:63];
    (* ram_style="distributed" *) reg signed [15:0] coarse [0:63];
    reg signed [15:0] class_logits [0:3];
    reg signed [15:0] size_p [0:1];
    reg signed [15:0] size_out [0:1];
    (* ram_style="distributed" *) reg signed [16:0] heat_mem [0:255];
    (* ram_style="distributed" *) reg [15:0] exp_mem [0:255];
    assign result_class_logits={class_logits[3],class_logits[2],class_logits[1],class_logits[0]};
    assign result_size={size_out[1],size_out[0]};

    reg [6:0] pool_channel=0;
    reg [5:0] pool_index=0;
    reg [13:0] pool_sum=0;
    reg [2:0] head_row=0;
    reg [7:0] hcell=0;          // 16x16 heat hcell, also 8x8 coarse hcell
    reg [5:0] att_channel=0;
    reg signed [16:0] heat_max=0;
    reg [24:0] exp_total=0;    // S = sum(e) < 2^24
    reg [29:0] num_x=0,num_y=0;
    reg [32:0] att_sum=0;
    reg [15:0] exp_d=0;

    // Dot-product subroutine: INT8 operands (pooled / att registers or bank memory at
    // base + k*stride) . INT8 weights + INT32 bias, requantised to INT16 Q8.8.
    reg [1:0] dot_src=0;       // 0 pooled, 1 att, 2 activation memory
    reg [15:0] dot_addr=0;
    reg [8:0] dot_stride=0;
    reg [6:0] dot_n=0,dot_k=0;
    reg [14:0] dot_woff=0;
    reg [9:0] dot_poff=0;
    reg [5:0] dot_return=0;
    reg signed [15:0] dot_result=0;
    reg signed [7:0] dot_operand=0;
    wire signed [7:0] dot_x=(dot_src==2) ? source_q:dot_operand;
    reg signed [7:0] dot_xr=0;
    wire signed [15:0] dot_product=$signed(dot_xr)*$signed(weight_q);

    // Restoring divider: quotient = dividend / divisor (unsigned), 48 iterations.
    reg [47:0] div_dividend=0,div_quotient=0;
    reg [24:0] div_divisor=0,div_remainder=0;
    reg [5:0] div_count=0;
    reg [5:0] div_return=0;
    wire [25:0] div_shifted={div_remainder,div_dividend[47]};

    // Heat hcell index -> coarse 8x8 hcell (nearest upsample): {gy>>1, gx>>1}.
    wire [5:0] coarse_index={hcell[7:5],hcell[3:1]};
    wire signed [16:0] heat_value=$signed(dot_result)+$signed(coarse[coarse_index]);
    wire signed [17:0] heat_difference=$signed(heat_max)-$signed(heat_mem[hcell]);
    wire [17:0] exp_index_raw=(heat_difference+18'sd4)>>>3;
    wire [4:0] odd_x={hcell[3:0],1'b1};
    wire [4:0] odd_y={hcell[7:4],1'b1};
    wire signed [16:0] size_sum=$signed(size_p[head_row[0]])+$signed(dot_result);
    // Pipeline registers: every memory-output multiply is registered before it is accumulated.
    reg signed [15:0] dot_term=0;
    reg [22:0] att_term=0;
    reg [15:0] exp_reg=0;
    reg [20:0] term_x=0,term_y=0;

    reg [1:0] winner;
    reg signed [15:0] maximum;
    // fpga_viz timing fix: the class logits are final thousands of cycles before ARGMAX, so the
    // 4-way compare is registered once (same result, same cycle count) to meet 100 MHz with the taps.
    reg [1:0] winner_q=0;
    always @(posedge clk) winner_q<=winner;
    integer j;
    always_comb begin
        winner=0;maximum=class_logits[0];
        for(j=1;j<4;j=j+1) if($signed(class_logits[j])>$signed(maximum)) begin maximum=class_logits[j];winner=j;end
    end

    always @(posedge clk) begin
        write_en<=0;done<=0;tap_we<=0;pool_tap_we<=0;att_tap_we<=0;
        layer_done_stage<=layer_done_pending;layer_done<=layer_done_stage;layer_done_pending<=0;
        if(mac_valid) begin accumulator<=accumulator+$signed(mac_term_d);mac_valid<=0;end
        if(op_valid) begin mac_term_d<=op_tap ? mac_product:16'sd0;mac_valid<=1;op_valid<=0;end
        if(rq_valid) begin
            write_en<=1;write_addr<=rq_addr;write_bank<=rq_bank;
            write_data<=saturate8(rq_biased,requant_shift);rq_valid<=0;
            tap_we<=1;tap_layer<=rq_layer;tap_channel<=rq_channel;tap_global<=rq_global;tap_y<=rq_y;tap_x<=rq_x;
        end
        if(rst) begin
            state<=IDLE;busy<=0;source_bank<=0;inference_cycles<=0;mac_valid<=0;op_valid<=0;rq_valid<=0;
        end else begin
            if(busy) inference_cycles<=inference_cycles+1;
            case(state)
                IDLE: if(start) begin
                    busy<=1;inference_cycles<=0;source_bank<=0;layer<=0;state<=LAYER;
                end
                LAYER: begin
                    out_channel<=0;state<=P_ADDR;
                    cfg_stride<=L_STRIDE[layer];cfg_kernel<=L_KERNEL[layer];cfg_pad<=L_PAD[layer];cfg_dw<=L_DEPTHWISE[layer];
                    cfg_in_hw<=L_INPUT_HW[layer];cfg_out_hw<=L_OUTPUT_HW[layer];cfg_in_plane<=L_INPUT_PLANE[layer];
                    cfg_out_plane<=L_OUTPUT_PLANE[layer];cfg_woff<=L_WEIGHT_OFFSET[layer];cfg_wstep<=L_WEIGHT_STEP[layer];
                    cfg_cin<=L_CIN[layer];cfg_cout<=L_COUT[layer];cfg_karea<=L_KERNEL_AREA[layer];cfg_poff<=L_PARAMETER_OFFSET[layer];
                    cfg_src_off<=L_SRC_OFFSET[layer];cfg_dst_off<=L_DST_OFFSET[layer];
                end
                P_ADDR: begin parameter_addr<=cfg_poff+out_channel;state<=P_WAIT;end
                P_WAIT: state<=P_LOAD;
                P_LOAD: begin
                    bias<=$signed(parameter_q[63:32]);requant_multiplier<=parameter_q[31:8];requant_shift<=parameter_q[7:0];
                    out_y<=0;out_x<=0;state<=BEGIN_PIXEL;
                end
                BEGIN_PIXEL: begin
                    accumulator<=bias;in_channel<=0;kernel_y<=0;kernel_x<=0;state<=C_ADDR;
                end
                C_ADDR: begin
                    tap_valid<=(tap_iy>=0 && tap_ix>=0 && tap_iy<$signed({2'b0,cfg_in_hw}) && tap_ix<$signed({2'b0,cfg_in_hw}));
                    a_chan<=tap_ic*cfg_in_plane;a_row<=tap_iy*$signed({1'b0,cfg_in_hw});a_col<=tap_ix;
                    w_out<=out_channel*cfg_wstep;w_in<=(cfg_dw ? 7'd0:in_channel)*cfg_karea;w_tap<=kernel_y*cfg_kernel+kernel_x;
                    state<=C_WAIT;
                end
                C_WAIT: begin
                    read_addr<=cfg_src_off+a_chan[15:0]+a_row[15:0]+{{6{a_col[9]}},a_col};
                    weight_addr<=cfg_woff+w_out+w_in+w_tap;state<=C_MAC;
                end
                C_MAC: state<=C_NEXT;
                C_NEXT: begin
                    op_a<=source_q;op_w<=weight_q;op_tap<=tap_valid;op_valid<=1;
                    if(kernel_x+1<cfg_kernel) begin kernel_x<=kernel_x+1;state<=C_ADDR;end
                    else begin
                        kernel_x<=0;
                        if(kernel_y+1<cfg_kernel) begin kernel_y<=kernel_y+1;state<=C_ADDR;end
                        else begin
                            kernel_y<=0;
                            if(!cfg_dw && in_channel+1<cfg_cin) begin in_channel<=in_channel+1;state<=C_ADDR;end
                            else state<=C_REQUANT;
                        end
                    end
                end
                C_REQUANT: begin
                    o_chan<=out_channel*cfg_out_plane;o_row<=out_y*cfg_out_hw;state<=C_DRAIN;
                end
                C_DRAIN: state<=C_WRITE;  // last tap's product is accumulated on this edge
                C_WRITE: begin
                    requant_product<=$signed(accumulator)*$signed({1'b0,requant_multiplier[22:0]});
                    rq_addr<=cfg_dst_off+o_chan[15:0]+o_row+out_x;rq_bank<=source_bank;state<=C_ADVANCE;
                    rq_layer<=layer;rq_channel<=out_channel;rq_global<=cfg_poff+out_channel;rq_y<=out_y;rq_x<=out_x;
                end
                C_ADVANCE: begin
                    rq_biased<=requant_product+round_bias(requant_product[63],requant_shift);rq_valid<=1;
                    if(out_x+1<cfg_out_hw) begin out_x<=out_x+1;state<=BEGIN_PIXEL;end
                    else begin
                        out_x<=0;
                        if(out_y+1<cfg_out_hw) begin out_y<=out_y+1;state<=BEGIN_PIXEL;end
                        else if(out_channel+1<cfg_cout) begin out_channel<=out_channel+1;state<=P_ADDR;end
                        else begin
                            source_bank<=~source_bank;completed_layer<=layer;layer_done_pending<=1;
                            if(layer==NUM_LAYERS-1) state<=G_INIT;
                            else begin layer<=layer+1;state<=LAYER;end
                        end
                    end
                end
                // ---- head. source_bank now selects bank B (final map @32768, tap @0).
                G_INIT: begin pool_channel<=0;pool_index<=0;pool_sum<=0;state<=G_ADDR;end
                G_ADDR: begin read_addr<=16'd32768+{3'b0,pool_channel,pool_index};state<=G_WAIT;end
                G_WAIT: state<=G_ACC;
                G_ACC: begin
                    pool_sum<=pool_sum+{6'b0,source_q};
                    if(pool_index==63) state<=G_STORE;
                    else begin pool_index<=pool_index+1;state<=G_ADDR;end
                end
                G_STORE: begin
                    pooled[pool_channel]<=(pool_sum+14'd32)>>6;pool_sum<=0;pool_index<=0;
                    pool_tap_we<=1;pool_tap_channel<=pool_channel;pool_tap_value<=(pool_sum+14'd32)>>6;
                    if(pool_channel==95) begin head_row<=0;state<=R_START;end
                    else begin pool_channel<=pool_channel+1;state<=G_ADDR;end
                end
                // Rows over pooled: class 0..3, presence, size_p 0..1 (contiguous weights/parameters).
                R_START: begin
                    dot_src<=0;dot_n<=96;dot_woff<=H_CLS_W+head_row*96;dot_poff<=H_CLS_P+head_row;
                    dot_return<=R_DONE;state<=D_P_ADDR;
                end
                R_DONE: begin
                    if(head_row<4) class_logits[head_row[1:0]]<=dot_result;
                    else if(head_row==4) result_presence<=dot_result;
                    else size_p[head_row-5]<=dot_result;
                    if(head_row==6) begin hcell<=0;state<=HC_START;end
                    else begin head_row<=head_row+1;state<=R_START;end
                end
                HC_START: begin
                    dot_src<=2;dot_addr<=16'd32768+hcell;dot_stride<=64;dot_n<=96;
                    dot_woff<=H_HEAT_COARSE_W;dot_poff<=H_HEAT_COARSE_P;dot_return<=HC_DONE;state<=D_P_ADDR;
                end
                HC_DONE: begin
                    coarse[hcell[5:0]]<=dot_result;
                    if(hcell==63) begin hcell<=0;state<=HF_START;end
                    else begin hcell<=hcell+1;state<=HC_START;end
                end
                HF_START: begin
                    dot_src<=2;dot_addr<={8'b0,hcell};dot_stride<=256;dot_n<=64;
                    dot_woff<=H_HEAT_FINE_W;dot_poff<=H_HEAT_FINE_P;dot_return<=HF_DONE;state<=D_P_ADDR;
                end
                HF_DONE: begin
                    heat_mem[hcell]<=heat_value;
                    if(hcell==0 || heat_value>heat_max) heat_max<=heat_value;
                    if(hcell==255) begin hcell<=0;state<=X_INIT;end
                    else begin hcell<=hcell+1;state<=HF_START;end
                end
                // Softmax weights: e = EXP[min((max-heat+4)>>3, 512)], S, sum(e*(2x+1)), sum(e*(2y+1)).
                X_INIT: begin exp_total<=0;num_x<=0;num_y<=0;state<=X_ADDR;end
                X_ADDR: begin exp_addr<=(exp_index_raw>512) ? 10'd512:exp_index_raw[9:0];state<=X_WAIT;end
                X_WAIT: state<=X_MUL;
                X_MUL: begin exp_reg<=exp_q;term_x<=exp_q*odd_x;term_y<=exp_q*odd_y;state<=X_ACC;end
                X_ACC: begin
                    exp_mem[hcell]<=exp_reg;exp_total<=exp_total+exp_reg;
                    att_tap_we<=1;att_tap_index<=hcell;att_tap_value<=exp_reg;
                    num_x<=num_x+term_x;num_y<=num_y+term_y;
                    if(hcell==255) begin hcell<=0;att_channel<=0;att_sum<=0;state<=A_ADDR;end
                    else begin hcell<=hcell+1;state<=X_ADDR;end
                end
                // Attended tap features: att[c] = (sum(e*tap[c]) + S/2) / S.
                A_CH: begin hcell<=0;att_sum<=0;state<=A_ADDR;end
                A_ADDR: begin read_addr<={2'b0,att_channel,hcell};exp_d<=exp_mem[hcell];state<=A_WAIT;end
                A_WAIT: state<=A_MUL;
                A_MUL: begin att_term<=exp_d*source_q[6:0];state<=A_ACC;end  // tap values are ReLU outputs, 0..127
                A_ACC: begin
                    att_sum<=att_sum+att_term;
                    if(hcell==255) state<=A_DIVSET;
                    else begin hcell<=hcell+1;state<=A_ADDR;end
                end
                A_DIVSET: begin
                    div_dividend<={15'b0,att_sum}+(exp_total>>1);
                    div_divisor<=exp_total;div_return<=A_DONE;state<=V_RUN;div_count<=48;div_remainder<=0;div_quotient<=0;
                end
                A_DONE: begin
                    att[att_channel]<=div_quotient[7:0];
                    if(att_channel==63) begin
                        div_dividend<={7'b0,num_x,11'b0}+(exp_total>>1);div_divisor<=exp_total;div_return<=CX_DONE;
                        state<=V_RUN;div_count<=48;div_remainder<=0;div_quotient<=0;
                    end else begin att_channel<=att_channel+1;state<=A_CH;end
                end
                CX_DONE: begin
                    result_cx<=div_quotient[15:0];
                    div_dividend<={7'b0,num_y,11'b0}+(exp_total>>1);div_divisor<=exp_total;div_return<=CY_DONE;
                    state<=V_RUN;div_count<=48;div_remainder<=0;div_quotient<=0;
                end
                CY_DONE: begin result_cy<=div_quotient[15:0];head_row<=0;state<=SA_START;end
                SA_START: begin
                    dot_src<=1;dot_n<=64;dot_woff<=H_SIZE_A_W+head_row[0]*64;dot_poff<=H_SIZE_A_P+head_row[0];
                    dot_return<=SA_DONE;state<=D_P_ADDR;
                end
                SA_DONE: begin
                    size_out[head_row[0]]<=(size_sum>17'sd32767) ? 16'sh7fff:((size_sum< -17'sd32768) ? 16'sh8000:size_sum[15:0]);
                    if(head_row[0]) state<=ARGMAX;
                    else begin head_row<=1;state<=SA_START;end
                end
                ARGMAX: begin result_class<=winner_q;state<=FINISH;end
                FINISH: begin busy<=0;done<=1;state<=IDLE;end
                // ---- dot-product subroutine
                D_P_ADDR: begin parameter_addr<=dot_poff;dot_k<=0;state<=D_P_WAIT;end
                D_P_WAIT: state<=D_P_LOAD;
                D_P_LOAD: begin
                    accumulator<=$signed(parameter_q[63:32]);requant_multiplier<=parameter_q[31:8];requant_shift<=parameter_q[7:0];
                    state<=D_ADDR;
                end
                D_ADDR: begin
                    weight_addr<=dot_woff+dot_k;read_addr<=dot_addr;
                    dot_operand<=(dot_src==0) ? pooled[dot_k]:$signed(att[dot_k[5:0]]);
                    state<=D_WAIT;
                end
                D_WAIT: state<=D_MAC;
                D_MAC: begin dot_xr<=dot_x;state<=D_MUL;end
                D_MUL: begin dot_term<=dot_product;state<=D_NEXT;end
                D_NEXT: begin
                    accumulator<=accumulator+$signed(dot_term);
                    if(dot_k+1==dot_n) state<=D_LAST;
                    else begin dot_k<=dot_k+1;dot_addr<=dot_addr+dot_stride;state<=D_ADDR;end
                end
                D_LAST: begin requant_product<=$signed(accumulator)*$signed({1'b0,requant_multiplier[22:0]});state<=D_RQ;end
                D_RQ: begin rq_biased<=requant_product+round_bias(requant_product[63],requant_shift);state<=D_STORE;end
                D_STORE: begin dot_result<=saturate16(rq_biased,requant_shift);state<=dot_return;end
                // ---- divider subroutine
                V_RUN: begin
                    div_dividend<=div_dividend<<1;div_count<=div_count-1;
                    if(div_shifted>={1'b0,div_divisor}) begin div_remainder<=div_shifted-{1'b0,div_divisor};div_quotient<={div_quotient[46:0],1'b1};end
                    else begin div_remainder<=div_shifted[24:0];div_quotient<={div_quotient[46:0],1'b0};end
                    if(div_count==1) state<=div_return;
                end
                default: begin state<=IDLE;busy<=0;end
            endcase
        end
    end
endmodule
