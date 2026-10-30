module uart_rx #(parameter integer CLOCK_HZ=100000000,BAUD=1000000)(
    input wire clk,rst,rx,
    output reg [7:0] data=0,output reg valid=0,output reg framing_error=0
);
    localparam integer TICKS=(CLOCK_HZ+BAUD/2)/BAUD;
    (* ASYNC_REG="TRUE" *) reg rx_meta=1,rx_sync=1;
    always @(posedge clk) begin rx_meta<=rx;rx_sync<=rx_meta;end
    reg [15:0] ticks=0;
    reg [3:0] bit_index=0;
    reg [1:0] state=0;
    reg [7:0] shift=0;
    always @(posedge clk) begin
        valid<=0;framing_error<=0;
        if(rst) begin state<=0;ticks<=0;end
        else case(state)
            0: if(!rx_sync) begin ticks<=TICKS/2-1;state<=1;end
            1: if(ticks==0) begin
                if(!rx_sync) begin ticks<=TICKS-1;bit_index<=0;state<=2;end else state<=0;
            end else ticks<=ticks-1;
            2: if(ticks==0) begin
                shift[bit_index]<=rx_sync;ticks<=TICKS-1;
                if(bit_index==7) state<=3;else bit_index<=bit_index+1;
            end else ticks<=ticks-1;
            3: if(ticks==0) begin
                if(rx_sync) begin data<=shift;valid<=1;end else framing_error<=1;
                state<=0;
            end else ticks<=ticks-1;
        endcase
    end
endmodule

module uart_tx #(parameter integer CLOCK_HZ=100000000,BAUD=1000000)(
    input wire clk,rst,start,input wire [7:0] data,
    output reg tx=1,output reg busy=0
);
    localparam integer TICKS=(CLOCK_HZ+BAUD/2)/BAUD;
    reg [15:0] ticks=0;
    reg [3:0] bit_index=0;
    reg [9:0] shift=10'h3ff;
    always @(posedge clk) begin
        if(rst) begin tx<=1;busy<=0;ticks<=0;end
        else if(!busy) begin
            tx<=1;
            if(start) begin shift<={1'b1,data,1'b0};tx<=0;busy<=1;ticks<=TICKS-1;bit_index<=0;end
        end else if(ticks==0) begin
            if(bit_index==9) begin busy<=0;tx<=1;end
            else begin shift<={1'b1,shift[9:1]};tx<=shift[1];bit_index<=bit_index+1;ticks<=TICKS-1;end
        end else ticks<=ticks-1;
    end
endmodule
