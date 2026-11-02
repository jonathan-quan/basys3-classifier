`timescale 1ns/1ps
// Full-system UART test of the fpga_viz design: every original packet must be
// byte-identical to fpga/export/vectors, and the new telemetry readout must match
// the integer reference (fpga_viz/make_vectors.py) byte for byte.
module tb_viz_uart;
    reg clk=0,rst=1,rx=1;
    wire tx;
    wire [15:0] led;
    always #5 clk=~clk;
    basys3_classifier_viz dut(.clk(clk),.btnC(rst),.RsRx(rx),.RsTx(tx),.led(led));
    reg [7:0] bytes[0:8191],packet[0:27669],expected[0:4095];
    reg [7:0] captured;
    integer received=0,base=0;
    initial forever begin
        @(negedge tx);#1500;
        for(integer bitn=0;bitn<8;bitn=bitn+1) begin captured[bitn]=tx;#1000;end
        if(tx!==1) $fatal(1,"UART TX stop bit invalid");
        bytes[received]=captured;received=received+1;
    end
    task send_byte(input [7:0] data);
        begin
            @(negedge clk);rx=0;repeat(100) @(negedge clk);
            for(integer bitn=0;bitn<8;bitn=bitn+1) begin rx=data[bitn];repeat(100) @(negedge clk);end
            rx=1;repeat(100) @(negedge clk);
        end
    endtask
    task check_reply(input string filename,input integer length);
        begin
            wait(received>=base+length);
            $readmemh(filename,expected,0,length-1);
            for(integer i=0;i<length;i=i+1) if(bytes[base+i]!==expected[i])
                $fatal(1,"UART packet mismatch %s byte %0d expected %02x actual %02x",filename,i,expected[i],bytes[base+i]);
            base=base+length;
            repeat(2000) @(negedge clk);
            if(received!=base) $fatal(1,"Unexpected extra bytes after %s",filename);
        end
    endtask
    initial begin
        repeat(20) @(negedge clk);rst=0;repeat(20) @(negedge clk);
        $readmemh("fpga/export/vectors/hello_request.mem",packet,0,17);
        for(integer i=0;i<18;i=i+1) send_byte(packet[i]);
        check_reply("fpga/export/vectors/hello_response.mem",34);$display("PASS HELLO unchanged");
        $readmemh("fpga_viz/vectors/telemetry_request_early.mem",packet,0,17);
        for(integer i=0;i<18;i=i+1) send_byte(packet[i]);
        check_reply("fpga_viz/vectors/telemetry_error_response.mem",34);$display("PASS telemetry refused before any inference");
        $readmemh("fpga/export/vectors/hello_request.mem",packet,0,17);
        packet[17]=packet[17]^8'h01;
        for(integer i=0;i<18;i=i+1) send_byte(packet[i]);
        check_reply("fpga/export/vectors/error_response.mem",34);$display("PASS corrupted CRC rejected");
        $readmemh("fpga/export/vectors/format_request.mem",packet,0,21);
        for(integer i=0;i<22;i=i+1) send_byte(packet[i]);
        check_reply("fpga/export/vectors/format_response.mem",34);$display("PASS invalid image configuration rejected");
        if(dut.core_busy) $fatal(1,"Invalid packet triggered neural inference");
        $readmemh("fpga/export/vectors/image_request.mem",packet);
        for(integer i=0;i<27670;i=i+1) send_byte(packet[i]);
        check_reply("fpga/export/vectors/image_response.mem",34);$display("PASS image result unchanged");
        if(led[15:8]===8'h00) $fatal(1,"Score LEDs did not update");
        $readmemh("fpga_viz/vectors/telemetry_request.mem",packet,0,17);
        for(integer i=0;i<18;i=i+1) send_byte(packet[i]);
        check_reply("fpga_viz/vectors/telemetry_response.mem",2858);
        $display("PASS telemetry: layer stamps, pooled vector, 256 channel sums and 864 heat bins match the integer reference");
        $display("ALL VIZ UART PACKETS MATCH");$finish;
    end
    initial begin #900000000;$fatal(1,"Viz UART simulation timeout");end
endmodule
