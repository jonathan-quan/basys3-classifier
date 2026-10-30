`timescale 1ns/1ps
module tb_det_core;
    `include "layers.svh"
    reg clk=0,rst=1,start=0,load_we=0;
    always #5 clk=~clk;
    reg [15:0] load_addr=0;
    reg [7:0] load_data=0;
    wire busy,done,layer_done;
    wire [63:0] class_logits;
    wire signed [15:0] presence;
    wire [15:0] cx,cy;
    wire [31:0] size;
    wire [1:0] result_class;
    wire [31:0] cycles;
    wire [3:0] completed_layer;
    detector_core dut(.clk(clk),.rst(rst),.start(start),.load_we(load_we),.load_addr(load_addr),.load_data(load_data),
        .busy(busy),.done(done),.result_class_logits(class_logits),.result_presence(presence),.result_cx(cx),.result_cy(cy),
        .result_size(size),.result_class(result_class),.inference_cycles(cycles),.layer_done(layer_done),.completed_layer(completed_layer));
    reg [7:0] image_data[0:49151],expected[0:65535],expected_pool[0:95],expected_att[0:63];
    reg [31:0] expected_heat[0:255];
    reg [15:0] expected_result[0:8],actual_result[0:8];
    integer test_case=0,i,count,offset;
    reg [7:0] actual;
    string name;
    always @(posedge clk) if(layer_done) begin
        count=L_COUT[completed_layer]*L_OUTPUT_PLANE[completed_layer];offset=L_DST_OFFSET[completed_layer];
        name=$sformatf("fpga_det/export/vectors/case%0d_layer%0d.mem",test_case,completed_layer);
        $readmemh(name,expected,0,count-1);
        for(integer k=0;k<count;k=k+1) begin
            actual=dut.source_bank ? dut.activation_b[offset+k]:dut.activation_a[offset+k];
            if(actual!==expected[k]) $fatal(1,"Case %0d layer %0d index %0d expected %0d actual %0d",test_case,completed_layer,k,$signed(expected[k]),$signed(actual));
        end
        $display("PASS case=%0d layer=%0d bytes=%0d cycle=%0d",test_case,completed_layer,count,cycles);
    end
    initial begin
        repeat(10) @(negedge clk);rst=0;
        for(test_case=0;test_case<5;test_case=test_case+1) begin
            name=$sformatf("fpga_det/export/vectors/input%0d.mem",test_case);$readmemh(name,image_data);
            for(i=0;i<49152;i=i+1) begin @(negedge clk);load_we=1;load_addr=i;load_data=image_data[i];end
            @(negedge clk);load_we=0;start=1;
            @(negedge clk);start=0;
            wait(done);#1;
            name=$sformatf("fpga_det/export/vectors/case%0d_pooled.mem",test_case);$readmemh(name,expected_pool);
            for(i=0;i<96;i=i+1) if(dut.pooled[i]!==expected_pool[i]) $fatal(1,"Pool mismatch case %0d index %0d",test_case,i);
            name=$sformatf("fpga_det/export/vectors/case%0d_heat.mem",test_case);$readmemh(name,expected_heat);
            for(i=0;i<256;i=i+1) if($signed(dut.heat_mem[i])!==$signed(expected_heat[i]))
                $fatal(1,"Heat mismatch case %0d cell %0d expected %0d actual %0d",test_case,i,$signed(expected_heat[i]),$signed(dut.heat_mem[i]));
            name=$sformatf("fpga_det/export/vectors/case%0d_att.mem",test_case);$readmemh(name,expected_att);
            for(i=0;i<64;i=i+1) if(dut.att[i]!==expected_att[i])
                $fatal(1,"Attention mismatch case %0d channel %0d expected %0d actual %0d",test_case,i,expected_att[i],dut.att[i]);
            name=$sformatf("fpga_det/export/vectors/case%0d_result.mem",test_case);$readmemh(name,expected_result);
            for(i=0;i<4;i=i+1) actual_result[i]=class_logits[i*16+:16];
            actual_result[4]=presence;actual_result[5]=cx;actual_result[6]=cy;actual_result[7]=size[15:0];actual_result[8]=size[31:16];
            for(i=0;i<9;i=i+1) if(actual_result[i]!==expected_result[i])
                $fatal(1,"Result mismatch case %0d output %0d expected %0d actual %0d",test_case,i,$signed(expected_result[i]),$signed(actual_result[i]));
            $display("PASS FINAL case=%0d class=%0d presence=%0d cx=%0d cy=%0d size=%0d,%0d cycles=%0d",test_case,result_class,
                presence,cx,cy,$signed(size[15:0]),$signed(size[31:16]),cycles);
            repeat(10) @(negedge clk);
        end
        $display("ALL DETECTOR LAYERS, POOL, HEAT, ATTENTION AND BOX OUTPUTS MATCH");$finish;
    end
    initial begin #(4.0e9);$fatal(1,"Simulation timeout");end
endmodule
