// Shared scalar MAC engine. CHW INT8 activations, INT32 bias/accumulation.
// Rounding, saturation, pooling and output scaling match fpga.quantize exactly.
module classifier_core #(
    parameter WEIGHTS_FILE="fpga/export/weights.mem",
    parameter PARAMETER_FILE="fpga/export/parameters.mem",
    parameter EXP_FILE="fpga/export/exp.mem"
)(
    input wire clk, rst, start,
    input wire load_we,
    input wire [14:0] load_addr,
    input wire [7:0] load_data,
    output reg busy=0, done=0,
    output reg [1:0] result_class=0,
    output reg [15:0] result_confidence=0,
    output wire [63:0] result_logits,
    output reg [31:0] inference_cycles=0,
    output reg layer_done=0,
    output reg [3:0] completed_layer=0
);
    `include "layers.svh"
    (* rom_style="block" *) reg signed [7:0] weights [0:8191];
    (* rom_style="block" *) reg [63:0] parameters [0:511];
    (* rom_style="block" *) reg [15:0] exp_table [0:512];
    initial begin
        $readmemh(WEIGHTS_FILE,weights);
        $readmemh(PARAMETER_FILE,parameters);
        $readmemh(EXP_FILE,exp_table);
    end

    reg source_bank=0;
    reg [14:0] read_addr=0,write_addr=0;
    reg [12:0] weight_addr=0;
    reg [8:0] parameter_addr=0;
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

    // Ping-pong activation banks. Each bank is one simple-dual-port block RAM:
    // exactly one write port and one registered read port (Vivado SDP template).
    // Bank A's write port is shared by image loading and inference through a
    // mux with the original priority (load while idle wins over inference write).
    // Inference never reads and writes the same bank in one layer, so there is
    // no same-address read/write collision during computation.
    wire loading=load_we && !busy;
    reg write_bank=0; // source bank of the layer that produced write_data
    wire bank_a_we=loading || (write_en && write_bank);
    wire bank_b_we=!loading && write_en && !write_bank;
    wire [14:0] bank_a_waddr=loading ? load_addr:write_addr;
    wire [7:0] bank_a_wdata=loading ? load_data:write_data;
    // Each bank uses the Vivado simple-dual-port template: one write process
    // with a single address/data/enable, and one separate registered read.
    // Arrays are never reset or initialised; every byte is written before use.
    (* ram_style="block" *) reg signed [7:0] activation_a [0:32767];
    (* ram_style="block" *) reg signed [7:0] activation_b [0:32767];
    reg signed [7:0] bank_a_q=0,bank_b_q=0;
    always @(posedge clk) if (bank_a_we) activation_a[bank_a_waddr]<=bank_a_wdata;
    always @(posedge clk) bank_a_q<=activation_a[read_addr];
    always @(posedge clk) if (bank_b_we) activation_b[write_addr]<=write_data;
    always @(posedge clk) bank_b_q<=activation_b[read_addr];
    // Bank select is captured on the same edge as the RAM read, so source_q is
    // bit- and cycle-identical to the previous single shared read register.
    reg read_bank=0;
    always @(posedge clk) read_bank<=source_bank;
    wire signed [7:0] source_q=read_bank ? bank_b_q:bank_a_q;

    localparam IDLE=0,LAYER=1,P_ADDR=2,P_WAIT=3,P_LOAD=4,BEGIN_PIXEL=5,
        C_ADDR=6,C_WAIT=7,C_MAC=8,C_NEXT=9,C_REQUANT=10,C_WRITE=11,C_ADVANCE=12,
        G_INIT=13,G_ADDR=14,G_WAIT=15,G_ACC=16,G_STORE=17,
        F_ADDR_PARAM=18,F_WAIT_PARAM=19,F_LOAD_PARAM=20,F_ADDR=21,F_WAIT=22,F_MAC=23,F_NEXT=24,
        F_REQUANT=25,F_STORE=26,ARGMAX=27,E_ADDR=28,E_WAIT=29,E_ACC=30,
        D_INIT=31,D_RUN=32,FINISH=33;
    reg [5:0] state=IDLE;
    reg [3:0] layer=0;
    reg [5:0] out_channel=0,in_channel=0;
    reg [5:0] out_y=0,out_x=0;
    reg [1:0] kernel_y=0,kernel_x=0;
    reg tap_valid=0;
    reg signed [31:0] bias=0,accumulator=0;
    reg [23:0] requant_multiplier=0;
    reg [7:0] requant_shift=0;
    (* use_dsp="yes" *) reg signed [63:0] requant_product=0;
    wire signed [15:0] mac_product=$signed(source_q)*$signed(weight_q);
    reg signed [7:0] pooled [0:47];
    reg [5:0] pool_channel=0,pool_index=0;
    reg [15:0] pool_sum=0;
    reg [1:0] fc_channel=0;
    reg [5:0] fc_index=0;
    reg signed [7:0] fc_activation=0;
    reg signed [15:0] logits [0:3];
    wire signed [15:0] fc_product=$signed(fc_activation)*$signed(weight_q);
    assign result_logits={logits[3],logits[2],logits[1],logits[0]};

    // Layer dimensions are copied from the generated layer table into registers
    // once per layer (LAYER state), so no datapath indexes the table by `layer`.
    reg [1:0] cfg_stride=1,cfg_kernel=1;
    reg cfg_pad=0,cfg_dw=0;
    reg [6:0] cfg_in_hw=0,cfg_out_hw=0;
    reg [13:0] cfg_in_plane=0;
    reg [11:0] cfg_out_plane=0;
    reg [12:0] cfg_woff=0;
    reg [5:0] cfg_wstep=0,cfg_cin=0,cfg_cout=0;
    reg [3:0] cfg_karea=0;
    reg [8:0] cfg_poff=0;

    // Two-stage tap address pipeline (C_ADDR: partial products, C_WAIT: sums).
    // Same values as the original single-cycle expressions for every valid tap.
    reg signed [8:0] tap_iy,tap_ix;
    reg [5:0] tap_ic;
    always_comb begin
        tap_iy=$signed({3'b0,(cfg_stride==2) ? {out_y,1'b0}:{1'b0,out_y}})+$signed({7'b0,kernel_y})-$signed({8'b0,cfg_pad});
        tap_ix=$signed({3'b0,(cfg_stride==2) ? {out_x,1'b0}:{1'b0,out_x}})+$signed({7'b0,kernel_x})-$signed({8'b0,cfg_pad});
        tap_ic=cfg_dw ? out_channel:in_channel;
    end
    reg [19:0] a_chan=0;
    reg signed [16:0] a_row=0;
    reg signed [8:0] a_col=0;
    reg [11:0] w_out=0;
    reg [9:0] w_in=0;
    reg [3:0] w_tap=0;
    reg [17:0] o_chan=0;
    reg [12:0] o_row=0;
    reg signed [15:0] mac_term_d=0;
    reg mac_valid=0;
    reg [16:0] pool_div=0;
    reg [5:0] pool_div_channel=0;
    reg pool_div_valid=0,rq_valid=0,layer_done_stage=0;
    localparam [15:0] POOL_RECIP=16'd58255;
    wire [32:0] pool_div_product=pool_div*POOL_RECIP; // floor(x/36) == (x*58255)>>21 for all x < 65554 (exhaustively checked)

    // Requantisation, split into two pipeline stages with no extra FSM cycles.
    // Reference (fpga.quantize.requant): r = sign(p)*((|p| + h) >> s), h = s ? 2^(s-1) : 0,
    // then saturation. Equivalent form used here (verified bit-exact, including all
    // halfway cases): r = (p + (p<0 ? h-1 : h)) >>> s.
    //   Stage 1: rq_biased = p + bias          (one 64-bit add)
    //   Stage 2: window/saturate rq_biased      (no 64-bit negate/compare chain)
    // r > T  <=>  rq_biased >= (T+1)<<s, tested by OR-ing the bits at and above s+k.
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
    reg [14:0] rq_addr=0;
    reg rq_bank=0,layer_done_pending=0;

    reg [1:0] winner,exp_channel=0;
    reg signed [15:0] maximum,max_logit=0;
    integer j,difference,lookup_index;
    always_comb begin
        winner=0;maximum=logits[0];
        for(j=1;j<4;j=j+1) if($signed(logits[j])>$signed(maximum)) begin
            maximum=logits[j];winner=j;
        end
        difference=$signed(max_logit)-$signed(logits[exp_channel]);
        lookup_index=(difference+4)>>3;
    end
    reg [18:0] exp_sum=0,denominator=0;
    reg [32:0] remainder=0;
    reg [31:0] dividend=0,quotient=0;
    reg [5:0] divide_count=0;
    wire [32:0] shifted_remainder={remainder[31:0],dividend[31]};

    always @(posedge clk) begin
        write_en<=0;done<=0;
        // layer_done follows the last pipelined RAM write of the layer.
        layer_done_stage<=layer_done_pending;layer_done<=layer_done_stage;layer_done_pending<=0;
        // Accumulate the previous tap's product (registered in C_NEXT).
        if(mac_valid) begin accumulator<=accumulator+$signed(mac_term_d);mac_valid<=0;end
        // Final requantisation stage and pipelined activation write.
        if(rq_valid) begin
            write_en<=1;write_addr<=rq_addr;write_bank<=rq_bank;
            write_data<=saturate8(rq_biased,requant_shift);rq_valid<=0;
        end
        // Pipelined global-average-pool division.
        if(pool_div_valid) begin
            pooled[pool_div_channel]<=pool_div_product[28:21];pool_div_valid<=0;
        end
        if(rst) begin
            state<=IDLE;busy<=0;source_bank<=0;inference_cycles<=0;mac_valid<=0;rq_valid<=0;pool_div_valid<=0;
            result_class<=0;result_confidence<=0;
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
                // Tap pipeline: C_ADDR partial address terms, C_WAIT address sums,
                // C_MAC RAM/ROM read, C_NEXT multiply, next edge accumulate.
                C_ADDR: begin
                    tap_valid<=(tap_iy>=0 && tap_ix>=0 && tap_iy<$signed({2'b0,cfg_in_hw}) && tap_ix<$signed({2'b0,cfg_in_hw}));
                    a_chan<=tap_ic*cfg_in_plane;a_row<=tap_iy*$signed({1'b0,cfg_in_hw});a_col<=tap_ix;
                    w_out<=out_channel*cfg_wstep;w_in<=(cfg_dw ? 6'd0:in_channel)*cfg_karea;w_tap<=kernel_y*cfg_kernel+kernel_x;
                    state<=C_WAIT;
                end
                C_WAIT: begin
                    read_addr<=a_chan[14:0]+a_row[14:0]+{{6{a_col[8]}},a_col};
                    weight_addr<=cfg_woff+w_out+w_in+w_tap;state<=C_MAC;
                end
                C_MAC: state<=C_NEXT;
                C_NEXT: begin
                    mac_term_d<=tap_valid ? mac_product:16'sd0;mac_valid<=1;
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
                // Requant pipeline (last tap is accumulated on the C_REQUANT edge):
                // C_WRITE multiply, C_ADVANCE rounding bias, next edge saturate+write,
                // following edge RAM write into the bank opposite the source bank.
                C_REQUANT: begin
                    o_chan<=out_channel*cfg_out_plane;o_row<=out_y*cfg_out_hw;state<=C_WRITE;
                end
                C_WRITE: begin
                    requant_product<=$signed(accumulator)*$signed({1'b0,requant_multiplier[22:0]});
                    rq_addr<=o_chan[14:0]+o_row[12:0]+out_x;rq_bank<=source_bank;state<=C_ADVANCE;
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
                            if(layer==8) state<=G_INIT;
                            else begin layer<=layer+1;state<=LAYER;end
                        end
                    end
                end
                G_INIT: begin pool_channel<=0;pool_index<=0;pool_sum<=0;state<=G_ADDR;end
                G_ADDR: begin read_addr<=pool_channel*36+pool_index;state<=G_WAIT;end
                G_WAIT: state<=G_ACC;
                G_ACC: begin
                    pool_sum<=pool_sum+source_q;
                    if(pool_index==35) state<=G_STORE;
                    else begin pool_index<=pool_index+1;state<=G_ADDR;end
                end
                G_STORE: begin
                    pool_div<=pool_sum+17'd18;pool_div_channel<=pool_channel;pool_div_valid<=1;
                    pool_sum<=0;pool_index<=0;
                    if(pool_channel==47) begin fc_channel<=0;state<=F_ADDR_PARAM;end
                    else begin pool_channel<=pool_channel+1;state<=G_ADDR;end
                end
                F_ADDR_PARAM: begin parameter_addr<=256+fc_channel;state<=F_WAIT_PARAM;end
                F_WAIT_PARAM: state<=F_LOAD_PARAM;
                F_LOAD_PARAM: begin
                    accumulator<=$signed(parameter_q[63:32]);requant_multiplier<=parameter_q[31:8];requant_shift<=parameter_q[7:0];
                    fc_index<=0;state<=F_ADDR;
                end
                F_ADDR: begin weight_addr<=5632+fc_channel*48+fc_index;fc_activation<=pooled[fc_index];state<=F_WAIT;end
                F_WAIT: state<=F_MAC;
                F_MAC: begin accumulator<=accumulator+$signed(fc_product);state<=F_NEXT;end
                F_NEXT: begin
                    // The FC accumulator is final here, so the requant multiply starts
                    // one state early, leaving F_REQUANT/F_STORE for the two rounding stages.
                    if(fc_index==47) begin
                        requant_product<=$signed(accumulator)*$signed({1'b0,requant_multiplier[22:0]});state<=F_REQUANT;
                    end
                    else begin fc_index<=fc_index+1;state<=F_ADDR;end
                end
                F_REQUANT: begin rq_biased<=requant_product+round_bias(requant_product[63],requant_shift);state<=F_STORE;end
                F_STORE: begin
                    logits[fc_channel]<=saturate16(rq_biased,requant_shift);
                    if(fc_channel==3) state<=ARGMAX;
                    else begin fc_channel<=fc_channel+1;state<=F_ADDR_PARAM;end
                end
                ARGMAX: begin
                    result_class<=winner;max_logit<=maximum;exp_channel<=0;exp_sum<=0;state<=E_ADDR;
                end
                E_ADDR: begin exp_addr<=lookup_index>512 ? 512:lookup_index;state<=E_WAIT;end
                E_WAIT: state<=E_ACC;
                E_ACC: begin
                    exp_sum<=exp_sum+exp_q;
                    if(exp_channel==3) state<=D_INIT;
                    else begin exp_channel<=exp_channel+1;state<=E_ADDR;end
                end
                D_INIT: begin
                    denominator<=exp_sum;remainder<=0;dividend<=32'hfffe0001+(exp_sum>>1);quotient<=0;divide_count<=32;state<=D_RUN;
                end
                D_RUN: begin
                    dividend<=dividend<<1;divide_count<=divide_count-1;
                    if(shifted_remainder>=denominator) begin remainder<=shifted_remainder-denominator;quotient<={quotient[30:0],1'b1};end
                    else begin remainder<=shifted_remainder;quotient<={quotient[30:0],1'b0};end
                    if(divide_count==1) state<=FINISH;
                end
                FINISH: begin result_confidence<=quotient[15:0];busy<=0;done<=1;state<=IDLE;end
                default: begin state<=IDLE;busy<=0;end
            endcase
        end
    end
endmodule
