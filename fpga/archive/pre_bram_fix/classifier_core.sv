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
    (* ram_style="block" *) reg signed [7:0] activation_a [0:32767];
    (* ram_style="block" *) reg signed [7:0] activation_b [0:32767];
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
    reg signed [7:0] write_data=0,source_q=0,weight_q=0;
    reg [63:0] parameter_q=0;
    reg [15:0] exp_q=0;
    always @(posedge clk) begin
        if (source_bank) source_q<=activation_b[read_addr];
        else source_q<=activation_a[read_addr];
        weight_q<=weights[weight_addr];
        parameter_q<=parameters[parameter_addr];
        exp_q<=exp_table[exp_addr];
        if (load_we && !busy) activation_a[load_addr]<=load_data;
        else if (write_en) begin
            if (source_bank) activation_a[write_addr]<=write_data;
            else activation_b[write_addr]<=write_data;
        end
    end

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

    // Address arithmetic has fixed layer dimensions, selected by a small ROM.
    integer iy,ix,ic,src_address,w_address;
    always @* begin
        iy=out_y*L_STRIDE[layer]+kernel_y-L_PAD[layer];
        ix=out_x*L_STRIDE[layer]+kernel_x-L_PAD[layer];
        ic=L_DEPTHWISE[layer] ? out_channel : in_channel;
        src_address=ic*L_INPUT_PLANE[layer]+iy*L_INPUT_HW[layer]+ix;
        w_address=L_WEIGHT_OFFSET[layer]+out_channel*L_WEIGHT_STEP[layer]
            +(L_DEPTHWISE[layer] ? 0:in_channel)*L_KERNEL_AREA[layer]+kernel_y*L_KERNEL[layer]+kernel_x;
    end

    function automatic signed [63:0] rounded(input signed [63:0] product,input [7:0] shift);
        reg [63:0] magnitude,value;
        begin
            magnitude=product[63] ? -product:product;
            value=(magnitude+((shift==0) ? 64'd0:(64'd1<<(shift-1))))>>shift;
            rounded=product[63] ? -$signed(value):$signed(value);
        end
    endfunction
    function automatic [7:0] quant8(input signed [63:0] product,input [7:0] shift);
        reg signed [63:0] value;
        begin
            value=rounded(product,shift);
            if (value<0) quant8=0;
            else if(value>127) quant8=127;
            else quant8=value[7:0];
        end
    endfunction
    function automatic [15:0] quant16(input signed [63:0] product,input [7:0] shift);
        reg signed [63:0] value;
        begin
            value=rounded(product,shift);
            if(value< -32768) quant16=16'h8000;
            else if(value>32767) quant16=16'h7fff;
            else quant16=value[15:0];
        end
    endfunction

    reg [1:0] winner,exp_channel=0;
    reg signed [15:0] maximum,max_logit=0;
    integer j,difference,lookup_index;
    always @* begin
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
        write_en<=0;done<=0;layer_done<=0;
        if(rst) begin
            state<=IDLE;busy<=0;source_bank<=0;inference_cycles<=0;
            result_class<=0;result_confidence<=0;
        end else begin
            if(busy) inference_cycles<=inference_cycles+1;
            case(state)
                IDLE: if(start) begin
                    busy<=1;inference_cycles<=0;source_bank<=0;layer<=0;state<=LAYER;
                end
                LAYER: begin out_channel<=0;state<=P_ADDR;end
                P_ADDR: begin parameter_addr<=L_PARAMETER_OFFSET[layer]+out_channel;state<=P_WAIT;end
                P_WAIT: state<=P_LOAD;
                P_LOAD: begin
                    bias<=$signed(parameter_q[63:32]);requant_multiplier<=parameter_q[31:8];requant_shift<=parameter_q[7:0];
                    out_y<=0;out_x<=0;state<=BEGIN_PIXEL;
                end
                BEGIN_PIXEL: begin
                    accumulator<=bias;in_channel<=0;kernel_y<=0;kernel_x<=0;state<=C_ADDR;
                end
                C_ADDR: begin
                    tap_valid<=(iy>=0 && ix>=0 && iy<L_INPUT_HW[layer] && ix<L_INPUT_HW[layer]);
                    read_addr<=src_address[14:0];weight_addr<=w_address[12:0];state<=C_WAIT;
                end
                C_WAIT: state<=C_MAC;
                C_MAC: begin if(tap_valid) accumulator<=accumulator+$signed(mac_product);state<=C_NEXT;end
                C_NEXT: begin
                    if(kernel_x+1<L_KERNEL[layer]) begin kernel_x<=kernel_x+1;state<=C_ADDR;end
                    else begin
                        kernel_x<=0;
                        if(kernel_y+1<L_KERNEL[layer]) begin kernel_y<=kernel_y+1;state<=C_ADDR;end
                        else begin
                            kernel_y<=0;
                            if(!L_DEPTHWISE[layer] && in_channel+1<L_CIN[layer]) begin in_channel<=in_channel+1;state<=C_ADDR;end
                            else state<=C_REQUANT;
                        end
                    end
                end
                C_REQUANT: begin
                    requant_product<=$signed(accumulator)*$signed({1'b0,requant_multiplier[22:0]});state<=C_WRITE;
                end
                C_WRITE: begin
                    write_en<=1;write_addr<=out_channel*L_OUTPUT_PLANE[layer]+out_y*L_OUTPUT_HW[layer]+out_x;
                    write_data<=quant8(requant_product,requant_shift);state<=C_ADVANCE;
                end
                C_ADVANCE: begin
                    if(out_x+1<L_OUTPUT_HW[layer]) begin out_x<=out_x+1;state<=BEGIN_PIXEL;end
                    else begin
                        out_x<=0;
                        if(out_y+1<L_OUTPUT_HW[layer]) begin out_y<=out_y+1;state<=BEGIN_PIXEL;end
                        else if(out_channel+1<L_COUT[layer]) begin out_channel<=out_channel+1;state<=P_ADDR;end
                        else begin
                            source_bank<=~source_bank;completed_layer<=layer;layer_done<=1;
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
                    pooled[pool_channel]<=(pool_sum+18)/36;
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
                    if(fc_index==47) state<=F_REQUANT;
                    else begin fc_index<=fc_index+1;state<=F_ADDR;end
                end
                F_REQUANT: begin requant_product<=$signed(accumulator)*$signed({1'b0,requant_multiplier[22:0]});state<=F_STORE;end
                F_STORE: begin
                    logits[fc_channel]<=quant16(requant_product,requant_shift);
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
