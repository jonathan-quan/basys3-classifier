`timescale 1ns/1ps
module tb_core;
    reg clk=0,rst=1,start=0,load_we=0;
    always #5 clk=~clk;
    reg [14:0] load_addr=0;
    reg [7:0] load_data=0;
    wire busy,done,layer_done;
    wire [1:0] result_class;
    wire [15:0] confidence;
    wire [63:0] logits;
    wire [31:0] cycles;
    wire [3:0] completed_layer;
    classifier_core dut(.clk(clk),.rst(rst),.start(start),.load_we(load_we),.load_addr(load_addr),.load_data(load_data),
        .busy(busy),.done(done),.result_class(result_class),.result_confidence(confidence),.result_logits(logits),
        .inference_cycles(cycles),.layer_done(layer_done),.completed_layer(completed_layer));
    reg [7:0] image_data[0:27647],expected[0:32767],expected_pool[0:47];
    reg [15:0] expected_logits[0:3],expected_conf[0:0];
    integer test_case=0,i,count,h,channels;
    reg [7:0] actual;
    string name;
    always @(posedge clk) if(layer_done) begin
        case(completed_layer)
            0:count=18432;
            1:count=4608;
            2:count=9216;
            3:count=2304;
            4:count=4608;
            5:count=1152;
            6,7,8:count=1728;
            default:$fatal(1,"Bad completed layer");
        endcase
        name=$sformatf("fpga/export/vectors/case%0d_layer%0d.mem",test_case,completed_layer);
        $readmemh(name,expected,0,count-1);
        for(integer k=0;k<count;k=k+1) begin
            actual=dut.source_bank ? dut.activation_b[k]:dut.activation_a[k];
            if(actual!==expected[k]) $fatal(1,"Case %0d layer %0d index %0d expected %0d actual %0d",test_case,completed_layer,k,$signed(expected[k]),$signed(actual));
        end
        $display("PASS case=%0d layer=%0d bytes=%0d",test_case,completed_layer,count);
    end
    initial begin
        repeat(10) @(negedge clk);rst=0;
        for(test_case=0;test_case<3;test_case=test_case+1) begin
            name=$sformatf("fpga/export/vectors/input%0d.mem",test_case);$readmemh(name,image_data);
            for(i=0;i<27648;i=i+1) begin @(negedge clk);load_we=1;load_addr=i;load_data=image_data[i];end
            @(negedge clk);load_we=0;start=1;
            @(negedge clk);start=0;
            wait(done);#1;
            name=$sformatf("fpga/export/vectors/pooled%0d.mem",test_case);$readmemh(name,expected_pool);
            for(i=0;i<48;i=i+1) if(dut.pooled[i]!==expected_pool[i]) $fatal(1,"Pool mismatch index %0d",i);
            name=$sformatf("fpga/export/vectors/logits%0d.mem",test_case);$readmemh(name,expected_logits);
            for(i=0;i<4;i=i+1) if(logits[i*16+:16]!==expected_logits[i]) $fatal(1,"FC mismatch case %0d class %0d expected %0d actual %0d",test_case,i,$signed(expected_logits[i]),$signed(logits[i*16+:16]));
            name=$sformatf("fpga/export/vectors/confidence%0d.mem",test_case);$readmemh(name,expected_conf);
            if(confidence!==expected_conf[0]) $fatal(1,"Confidence mismatch %0d %0d",confidence,expected_conf[0]);
            $display("PASS FINAL case=%0d class=%0d confidence=%0d cycles=%0d",test_case,result_class,confidence,cycles);
            repeat(10) @(negedge clk);
        end
        $display("ALL INTEGER LAYERS, POOL, LOGITS AND CONFIDENCE MATCH");$finish;
    end
    initial begin #250000000;$fatal(1,"Simulation timeout");end
endmodule
