// UART webcam-image classifier. All CNN inference, class selection and class
// softmax score execute on FPGA. Host performs capture/preprocessing/display.
// fpga_viz version: every packet of the original protocol is byte-identical. Added:
//   * request 0x20 TELEMETRY -> response 0xA0 with a 2,840-byte payload: the frame ID
//     of the last completed inference followed by inference_telemetry's 2,836 bytes
//     (error 0xFF code 4 if no inference has completed since reset/start).
//   * LED8-15: bar graph of the last class score (8 LEDs = 100%).
module basys3_classifier_viz #(parameter integer CLOCK_HZ=100000000,BAUD=1000000)(
    input wire clk,btnC,RsRx,
    output wire RsTx,
    output wire [15:0] led
);
    `include "hardware_id.svh"
    (* ASYNC_REG="TRUE" *) reg reset_meta=0,reset_sync=0;
    always @(posedge clk) begin reset_meta<=btnC;reset_sync<=reset_meta;end
    wire rst=reset_sync;
    wire [7:0] rx_data;
    wire rx_valid,rx_error,tx_busy;
    reg tx_start=0;
    reg [7:0] tx_data=0;
    uart_rx #(.CLOCK_HZ(CLOCK_HZ),.BAUD(BAUD)) u_rx(.clk(clk),.rst(rst),.rx(RsRx),.data(rx_data),.valid(rx_valid),.framing_error(rx_error));
    uart_tx #(.CLOCK_HZ(CLOCK_HZ),.BAUD(BAUD)) u_tx(.clk(clk),.rst(rst),.start(tx_start),.data(tx_data),.tx(RsTx),.busy(tx_busy));

    reg core_start=0,load_we=0;
    reg [14:0] load_addr=0;
    reg [7:0] load_data=0;
    wire core_busy,core_done;
    wire [1:0] core_class;
    wire [15:0] core_confidence;
    wire [63:0] core_logits;
    wire [31:0] core_cycles;
    wire core_layer_done,tap_we,pool_tap_we;
    wire [3:0] core_completed_layer,tap_layer;
    wire [7:0] tap_data,pool_tap_value;
    wire [5:0] tap_channel,tap_y,tap_x,pool_tap_channel;
    wire [8:0] tap_global;
    classifier_core u_core(.clk(clk),.rst(rst),.start(core_start),.load_we(load_we),.load_addr(load_addr),.load_data(load_data),
        .busy(core_busy),.done(core_done),.result_class(core_class),.result_confidence(core_confidence),
        .result_logits(core_logits),.inference_cycles(core_cycles),.layer_done(core_layer_done),.completed_layer(core_completed_layer),
        .tap_we(tap_we),.tap_data(tap_data),.tap_layer(tap_layer),.tap_channel(tap_channel),.tap_global(tap_global),
        .tap_y(tap_y),.tap_x(tap_x),.pool_tap_we(pool_tap_we),.pool_tap_channel(pool_tap_channel),.pool_tap_value(pool_tap_value));

    // Observation-only telemetry of the most recent inference.
    localparam [11:0] TELEMETRY_PAYLOAD=12'd2840; // 4-byte frame ID + 2,836 telemetry bytes
    reg [11:0] telemetry_addr=0;
    wire [7:0] telemetry_byte;
    inference_telemetry u_tel(.clk(clk),.tap_we(tap_we),.tap_data(tap_data),.tap_layer(tap_layer),.tap_channel(tap_channel),
        .tap_global(tap_global),.tap_y(tap_y),.tap_x(tap_x),.pool_we(pool_tap_we),.pool_channel(pool_tap_channel),
        .pool_value(pool_tap_value),.layer_done(core_layer_done),.completed_layer(core_completed_layer),.cycles(core_cycles),
        .read_addr(telemetry_addr),.read_data(telemetry_byte));
    reg telemetry_valid=0;

    function automatic [31:0] crc_byte(input [31:0] initial_crc,input [7:0] data);
        reg [31:0] c;
        integer i;
        begin
            c=initial_crc^data;
            for(i=0;i<8;i=i+1) c=c[0] ? (c>>1)^32'hedb88320:(c>>1);
            crc_byte=c;
        end
    endfunction

    localparam R_SYNC=0,R_HEADER=1,R_PAYLOAD=2,R_CRC=3;
    reg [1:0] rstate=R_SYNC;
    reg [1:0] sync_index=0,crc_index=0,pixel_channel=0;
    reg [3:0] header_index=0;
    reg [7:0] rx_version=0,rx_type=0;
    reg [31:0] frame_id=0,core_frame_id=0,payload_length=0,payload_index=0;
    reg [31:0] rx_crc=32'hffffffff,received_crc=0;
    reg [13:0] pixel_index=0;
    reg configuration_ok=0;
    reg [25:0] rx_timeout=0;
    reg error_seen=0;
    reg [3:0] class_leds=0;
    reg [26:0] heartbeat=0;
    always @(posedge clk) heartbeat<=heartbeat+1;
    reg [7:0] score_leds=0;
    wire [19:0] score_scaled=core_confidence*4'd9;
    assign led={score_leds,heartbeat[26],(rstate!=R_SYNC),error_seen,core_busy,class_leds};

    reg response_request=0;
    reg [7:0] response_type=0;
    reg [31:0] response_frame=0;
    reg [127:0] response_payload=0;
    localparam T_IDLE=0,T_SEND=1,T_LAUNCH=2,T_WAIT=3;
    reg [1:0] tstate=T_IDLE;
    wire transmitter_active=(tstate!=T_IDLE);
    reg [11:0] tx_index=0;
    reg [7:0] packet_type=0;
    reg [31:0] packet_frame=0,tx_crc=32'hffffffff;
    reg [127:0] packet_payload=0;
    reg packet_long=0;
    reg [11:0] packet_length=12'd16,crc_start=12'd30;

    // Header bytes 0-13, payload bytes 14..crc_start-1, CRC bytes crc_start..+3.
    // For 16-byte packets this produces exactly the original byte sequence.
    wire [11:0] payload_offset=tx_index-12'd14;
    wire [1:0] word_byte=tx_index[1:0]-2'd2;      // bytes 6-9 and 10-13 -> 0..3
    wire [11:0] crc_offset=tx_index-crc_start;
    wire [31:0] length_word={20'd0,packet_length};
    reg [7:0] next_tx_byte;
    always @* begin
        if(tx_index==0) next_tx_byte=8'hb3;
        else if(tx_index==1) next_tx_byte=8'h5d;
        else if(tx_index==2) next_tx_byte=8'ha7;
        else if(tx_index==3) next_tx_byte=8'hc1;
        else if(tx_index==4) next_tx_byte=8'd1;
        else if(tx_index==5) next_tx_byte=packet_type;
        else if(tx_index<10) next_tx_byte=packet_frame[word_byte*8+:8];
        else if(tx_index<14) next_tx_byte=length_word[word_byte*8+:8];
        else if(tx_index<crc_start) next_tx_byte=(packet_long && payload_offset>=12'd4) ? telemetry_byte:packet_payload[payload_offset[3:0]*8+:8];
        else next_tx_byte=(~tx_crc)>>(crc_offset[1:0]*8);
    end
    always @(posedge clk) begin
        tx_start<=0;
        if(rst) begin tstate<=T_IDLE;tx_index<=0;end
        else case(tstate)
            T_IDLE: if(response_request) begin
                packet_type<=response_type;packet_frame<=response_frame;packet_payload<=response_payload;
                packet_long<=(response_type==8'ha0);
                packet_length<=(response_type==8'ha0) ? TELEMETRY_PAYLOAD:12'd16;
                crc_start<=(response_type==8'ha0) ? TELEMETRY_PAYLOAD+12'd14:12'd30;
                tx_index<=0;tx_crc<=32'hffffffff;tstate<=T_SEND;
            end
            T_SEND: if(!tx_busy) begin
                tx_data<=next_tx_byte;tx_start<=1;
                if(tx_index>=4 && tx_index<crc_start) tx_crc<=crc_byte(tx_crc,next_tx_byte);
                tstate<=T_LAUNCH;
            end
            // Prefetch the telemetry byte for the NEXT payload position a full UART
            // character (~1,000 cycles) before it is needed (readout latency: 3 cycles).
            T_LAUNCH: begin telemetry_addr<=tx_index-12'd17;tstate<=T_WAIT;end
            T_WAIT:if(!tx_busy) begin
                if(tx_index==crc_start+12'd3) tstate<=T_IDLE;
                else begin tx_index<=tx_index+1;tstate<=T_SEND;end
            end
        endcase
    end

    always @(posedge clk) begin
        core_start<=0;load_we<=0;response_request<=0;
        if(rst) begin
            rstate<=R_SYNC;sync_index<=0;rx_timeout<=0;error_seen<=0;class_leds<=0;score_leds<=0;telemetry_valid<=0;
        end else begin
            if(rx_error) error_seen<=1;
            if(core_done) begin
                response_request<=1;response_type<=8'h91;response_frame<=core_frame_id;
                response_payload<={core_cycles,core_logits,core_confidence,8'h00,{6'b0,core_class}};
                class_leds<=4'b0001<<core_class;
                score_leds<=~(8'hff<<score_scaled[19:16]);
                telemetry_valid<=1;
            end
            if(rstate!=R_SYNC && !rx_valid) begin
                rx_timeout<=rx_timeout+1;
                if(rx_timeout==50000000) begin rstate<=R_SYNC;sync_index<=0;error_seen<=1;end
            end else rx_timeout<=0;
            // Stop-and-wait: never overwrite active inference memories.
            if(core_busy || transmitter_active || response_request) begin rstate<=R_SYNC;sync_index<=0;end
            else if(rx_valid) case(rstate)
                R_SYNC: begin
                    case(sync_index)
                        0:sync_index<=rx_data==8'hb3 ? 1:0;
                        1:sync_index<=rx_data==8'h5d ? 2:(rx_data==8'hb3 ? 1:0);
                        2:sync_index<=rx_data==8'ha7 ? 3:(rx_data==8'hb3 ? 1:0);
                        3:if(rx_data==8'hc1) begin
                            sync_index<=0;header_index<=0;frame_id<=0;payload_length<=0;
                            rx_crc<=32'hffffffff;rstate<=R_HEADER;
                        end else sync_index<=rx_data==8'hb3 ? 1:0;
                    endcase
                end
                R_HEADER:begin
                    rx_crc<=crc_byte(rx_crc,rx_data);header_index<=header_index+1;
                    case(header_index)
                        0:rx_version<=rx_data;
                        1:rx_type<=rx_data;
                        2,3,4,5:frame_id[(header_index-2)*8+:8]<=rx_data;
                        6,7,8,9:payload_length[(header_index-6)*8+:8]<=rx_data;
                    endcase
                    if(header_index==9) begin
                        payload_index<=0;pixel_index<=0;pixel_channel<=0;crc_index<=0;received_crc<=0;configuration_ok<=1;
                        if({rx_data,payload_length[23:0]}>27652) begin rstate<=R_SYNC;error_seen<=1;end
                        else if({rx_data,payload_length[23:0]}==0) rstate<=R_CRC;
                        else rstate<=R_PAYLOAD;
                    end
                end
                R_PAYLOAD:begin
                    rx_crc<=crc_byte(rx_crc,rx_data);payload_index<=payload_index+1;
                    if(rx_type==8'h10) begin
                        case(payload_index)
                            0:if(rx_data!=96) configuration_ok<=0;
                            1:if(rx_data!=0) configuration_ok<=0;
                            2:if(rx_data!=1) configuration_ok<=0;
                            3:if(rx_data!=0) configuration_ok<=0;
                            default:begin
                                load_we<=1;load_addr<=pixel_channel*9216+pixel_index;load_data<=rx_data^8'h80;
                                if(pixel_channel==2) begin pixel_channel<=0;pixel_index<=pixel_index+1;end
                                else pixel_channel<=pixel_channel+1;
                            end
                        endcase
                    end
                    if(payload_index+1==payload_length) begin rstate<=R_CRC;crc_index<=0;end
                end
                R_CRC:begin
                    received_crc[crc_index*8+:8]<=rx_data;crc_index<=crc_index+1;
                    if(crc_index==3) begin
                        rstate<=R_SYNC;sync_index<=0;response_frame<=frame_id;
                        if({rx_data,received_crc[23:0]}!=(~rx_crc)) begin
                            error_seen<=1;response_request<=1;response_type<=8'hff;response_payload<=128'd1;
                        end else if(rx_version!=1) begin
                            response_request<=1;response_type<=8'hff;response_payload<=128'd2;
                        end else if(rx_type==1 && payload_length==0) begin
                            response_request<=1;response_type<=8'h81;response_payload<={MODEL_ID,8'd1,8'd4,16'd96,32'h31433342};
                        end else if(rx_type==8'h10 && payload_length==27652 && configuration_ok) begin
                            core_frame_id<=frame_id;core_start<=1;telemetry_valid<=0;
                        end else if(rx_type==8'h20 && payload_length==0) begin
                            response_request<=1;
                            if(telemetry_valid) begin response_type<=8'ha0;response_payload<={96'd0,core_frame_id};end
                            else begin response_type<=8'hff;response_payload<=128'd4;end
                        end else begin response_request<=1;response_type<=8'hff;response_payload<=128'd3;end
                    end
                end
            endcase
        end
    end
endmodule
