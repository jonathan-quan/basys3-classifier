`timescale 1ns/1ps
module tb_det_uart;
    reg clk=0,rst=1,rx=1;
    wire tx;
    wire [7:0] led;
    always #5 clk=~clk;
    basys3_detector dut(.clk(clk),.btnC(rst),.RsRx(rx),.RsTx(tx),.led(led));
    reg [7:0] bytes[0:255],packet[0:49173],expected[0:41];
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
        $readmemh("fpga_det/export/vectors/hello_request.mem",packet,0,17);
        for(integer i=0;i<18;i=i+1) send_byte(packet[i]);
        check_reply("fpga_det/export/vectors/hello_response.mem",34);$display("PASS UART HELLO identity and CRC");
        packet[17]=packet[17]^8'h01;
        for(integer i=0;i<18;i=i+1) send_byte(packet[i]);
        check_reply("fpga_det/export/vectors/error_response.mem",34);$display("PASS UART corrupted CRC rejected");
        $readmemh("fpga_det/export/vectors/format_request.mem",packet,0,21);
        for(integer i=0;i<22;i=i+1) send_byte(packet[i]);
        check_reply("fpga_det/export/vectors/format_response.mem",34);$display("PASS UART invalid image configuration rejected");
        if(dut.core_busy) $fatal(1,"Invalid packet triggered neural inference");
        $readmemh("fpga_det/export/vectors/image_request.mem",packet);
        for(integer i=0;i<49174;i=i+1) send_byte(packet[i]);
        check_reply("fpga_det/export/vectors/image_response.mem",42);
        $display("ALL UART FRAMING, CRC, RGB TO CHW, DETECTION AND OUTPUT BYTES MATCH");$finish;
    end
    initial begin #(2.0e9);$fatal(1,"UART integration simulation timeout");end
endmodule
