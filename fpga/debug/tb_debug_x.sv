`timescale 1ns/1ps
// Diagnostic only: loads validation case 0 and reports the first unknown (X/Z)
// values seen inside the core datapath under 4-state XSim semantics.
module tb_debug_x;
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
    reg [7:0] image_data[0:27647];
    integer i,reports=0,checked=0,bad=0;
    always @(posedge clk) if(busy && reports<40) begin
        if($isunknown(dut.state)||$isunknown(dut.read_addr)||$isunknown(dut.weight_addr)||$isunknown(dut.tap_valid)
           ||$isunknown(dut.accumulator)||$isunknown(dut.mac_term)||$isunknown(dut.parameter_q)||$isunknown(dut.weight_q)
           ||(dut.state==8 && dut.tap_valid && $isunknown(dut.source_q))||$isunknown(dut.read_bank)||$isunknown(dut.source_bank)
           ||(dut.write_en && ($isunknown(dut.write_data)||$isunknown(dut.write_addr)))) begin
            reports=reports+1;
            $display("X t=%0t state=%0d layer=%0d oc=%0d ic=%0d y=%0d x=%0d ky=%0d kx=%0d tap=%b ra=%h wa=%h a_q=%h b_q=%h rb=%b sb=%b src=%h w=%h wq=%h pq=%h acc=%h mt=%h we=%b wd=%h wad=%h",
                $time,dut.state,dut.layer,dut.out_channel,dut.in_channel,dut.out_y,dut.out_x,dut.kernel_y,dut.kernel_x,dut.tap_valid,
                dut.read_addr,dut.weight_addr,dut.bank_a_q,dut.bank_b_q,dut.read_bank,dut.source_bank,dut.source_q,dut.weight_addr,dut.weight_q,
                dut.parameter_q,dut.accumulator,dut.mac_term,dut.write_en,dut.write_data,dut.write_addr);
        end
    end
    initial begin
        repeat(10) @(negedge clk);rst=0;
        $readmemh("fpga/export/vectors/input0.mem",image_data);
        for(i=0;i<27648;i=i+1) begin @(negedge clk);load_we=1;load_addr=i;load_data=image_data[i];end
        @(negedge clk);load_we=0;
        for(i=0;i<27648;i=i+1) begin
            checked=checked+1;
            if(dut.activation_a[i]!==image_data[i]) begin bad=bad+1; if(bad<6) $display("LOADCHECK addr=%0d expected=%h actual=%h",i,image_data[i],dut.activation_a[i]); end
        end
        $display("LOADCHECK checked=%0d bad=%0d busy=%b loading=%b",checked,bad,busy,dut.loading);
        start=1;@(negedge clk);start=0;
        wait(layer_done);#1;
        $display("LAYER0 out[0..3]=%h %h %h %h",dut.activation_b[0],dut.activation_b[1],dut.activation_b[2],dut.activation_b[3]);
        $display("DEBUG_X_DONE reports=%0d",reports);$finish;
    end
endmodule
