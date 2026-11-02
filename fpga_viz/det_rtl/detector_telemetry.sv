// Observation-only telemetry for the box detector (recording demo). For the most recent
// inference it keeps:
//   * the cycle count at which each of the 11 backbone layers finished     (11 x u32)
//   * the 96 global-average-pooled values                                 (96 x s8)
//   * the sum of every output channel, 704 channels in total              (704 x u32, 20 bits used)
//   * a coarse spatial heat map per layer, summed over channels           (1,664 x u16)
//       layers 0-4: 16x16 bins (4x4, 2x2, 2x2, 1x1, 1x1 pixels); layers 5-10: 8x8 (1x1)
//   * the softmax weight of every cell of the head's 16x16 box heat map   (256 x u16)
// Nothing here feeds back into the computation. Layout (little endian, 6,796 bytes):
//   0 stamps | 44 pooled | 140 channel sums | 2956 heat bins | 6284 attention
module detector_telemetry(
    input wire clk,
    input wire tap_we,
    input wire [7:0] tap_data,
    input wire [3:0] tap_layer,
    input wire [6:0] tap_channel,
    input wire [9:0] tap_global,
    input wire [5:0] tap_y,tap_x,
    input wire pool_we,
    input wire [6:0] pool_channel,
    input wire [7:0] pool_value,
    input wire att_we,
    input wire [7:0] att_index,
    input wire [15:0] att_value,
    input wire layer_done,
    input wire [3:0] completed_layer,
    input wire [31:0] cycles,
    input wire [12:0] read_addr,
    output reg [7:0] read_data=0
);
    localparam [12:0] POOL_BASE=13'd44,SUM_BASE=13'd140,HEAT_BASE=13'd2956,ATT_BASE=13'd6284;

    // ---- Layer finish times -------------------------------------------------
    reg [31:0] stamp [0:10];
    integer s;
    initial for(s=0;s<11;s=s+1) stamp[s]=0;
    always @(posedge clk) if(layer_done && completed_layer<=4'd10) stamp[completed_layer]<=cycles;

    // ---- Pooled vector --------------------------------------------------------
    reg [7:0] pooled [0:95];
    always @(posedge clk) if(pool_we) pooled[pool_channel]<=pool_value;

    // ---- Per-channel sums (channels are written consecutively, CHW order) ------
    (* ram_style="block" *) reg [19:0] channel_sum [0:1023];
    reg [19:0] running=0;
    wire [19:0] running_next=(tap_y==0 && tap_x==0) ? {12'd0,tap_data} : running+{12'd0,tap_data};
    reg sum_we=0;
    reg [9:0] sum_waddr=0;
    reg [19:0] sum_wdata=0;
    always @(posedge clk) begin
        sum_we<=tap_we;
        if(tap_we) begin running<=running_next;sum_waddr<=tap_global;sum_wdata<=running_next;end
        if(sum_we) channel_sum[sum_waddr]<=sum_wdata;
    end

    // ---- Spatial heat maps (read-modify-write) ----------------------------------
    reg [1:0] h_shift;
    reg h_grid16;
    reg [10:0] h_base;
    always_comb begin
        case(tap_layer)
            4'd0: begin h_shift=2'd2;h_grid16=1'b1;h_base=11'd0;   end
            4'd1: begin h_shift=2'd1;h_grid16=1'b1;h_base=11'd256; end
            4'd2: begin h_shift=2'd1;h_grid16=1'b1;h_base=11'd512; end
            4'd3: begin h_shift=2'd0;h_grid16=1'b1;h_base=11'd768; end
            4'd4: begin h_shift=2'd0;h_grid16=1'b1;h_base=11'd1024;end
            4'd5: begin h_shift=2'd0;h_grid16=1'b0;h_base=11'd1280;end
            4'd6: begin h_shift=2'd0;h_grid16=1'b0;h_base=11'd1344;end
            4'd7: begin h_shift=2'd0;h_grid16=1'b0;h_base=11'd1408;end
            4'd8: begin h_shift=2'd0;h_grid16=1'b0;h_base=11'd1472;end
            4'd9: begin h_shift=2'd0;h_grid16=1'b0;h_base=11'd1536;end
            default: begin h_shift=2'd0;h_grid16=1'b0;h_base=11'd1600;end
        endcase
    end
    wire [5:0] bin_y=tap_y>>h_shift,bin_x=tap_x>>h_shift;
    wire [5:0] bin_mask=(6'd1<<h_shift)-6'd1;
    wire [10:0] bin_index=h_base+(h_grid16 ? {1'b0,bin_y,4'b0} : {2'b0,bin_y,3'b0})+{5'd0,bin_x};
    wire first_in_bin=(tap_channel==0) && ((tap_y&bin_mask)==0) && ((tap_x&bin_mask)==0);

    (* ram_style="block" *) reg [15:0] heat [0:2047];
    reg h1_valid=0,h1_first=0,h2_valid=0,h2_first=0;
    reg [10:0] h1_bin=0,h2_bin=0;
    reg [7:0] h1_data=0,h2_data=0;
    reg [15:0] heat_q=0;
    reg [10:0] read_heat_index=0;
    wire [10:0] heat_raddr=h1_valid ? h1_bin:read_heat_index; // readout only happens between inferences
    always @(posedge clk) begin
        h1_valid<=tap_we;h1_bin<=bin_index;h1_first<=first_in_bin;h1_data<=tap_data;
        heat_q<=heat[heat_raddr];
        h2_valid<=h1_valid;h2_bin<=h1_bin;h2_first<=h1_first;h2_data<=h1_data;
        if(h2_valid) heat[h2_bin]<=h2_first ? {8'd0,h2_data} : heat_q+{8'd0,h2_data};
    end

    // ---- Head attention (softmax weights of the 16x16 heat map) -----------------
    (* ram_style="block" *) reg [15:0] attention [0:255];
    reg [7:0] read_att_index=0;
    reg [15:0] att_q=0;
    always @(posedge clk) begin
        if(att_we) attention[att_index]<=att_value;
        att_q<=attention[read_att_index];
    end

    // ---- Byte readout (3-cycle latency; the UART holds each address ~1,000 cycles) --
    reg [2:0] region=0,region_q=0;
    reg [1:0] byte_select=0,byte_select_q=0;
    reg [6:0] small_index=0;
    reg [9:0] read_sum_index=0;
    reg [31:0] stamp_q=0;
    reg [7:0] pooled_q=0;
    reg [19:0] sum_q=0;
    wire [12:0] pool_offset=read_addr-POOL_BASE,sum_offset=read_addr-SUM_BASE;
    wire [12:0] heat_offset=read_addr-HEAT_BASE,att_offset=read_addr-ATT_BASE;
    always @(posedge clk) begin
        if(read_addr<POOL_BASE) begin region<=3'd0;small_index<={3'b0,read_addr[5:2]};byte_select<=read_addr[1:0];end
        else if(read_addr<SUM_BASE) begin region<=3'd1;small_index<=pool_offset[6:0];byte_select<=2'd0;end
        else if(read_addr<HEAT_BASE) begin region<=3'd2;read_sum_index<=sum_offset[11:2];byte_select<=sum_offset[1:0];end
        else if(read_addr<ATT_BASE) begin region<=3'd3;read_heat_index<=heat_offset[11:1];byte_select<={1'b0,heat_offset[0]};end
        else begin region<=3'd4;read_att_index<=att_offset[8:1];byte_select<={1'b0,att_offset[0]};end
        stamp_q<=stamp[small_index<11 ? small_index[3:0]:4'd0];
        pooled_q<=pooled[small_index<96 ? small_index:7'd0];
        sum_q<=channel_sum[read_sum_index];
        region_q<=region;byte_select_q<=byte_select;
        case(region_q)
            3'd0: read_data<=stamp_q[byte_select_q*8+:8];
            3'd1: read_data<=pooled_q;
            3'd2: read_data<={12'd0,sum_q}>>(byte_select_q*8);
            3'd3: read_data<=byte_select_q[0] ? heat_q[15:8]:heat_q[7:0];
            default: read_data<=byte_select_q[0] ? att_q[15:8]:att_q[7:0];
        endcase
    end
endmodule
