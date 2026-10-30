// Generated from the preserved classifier. Do not hand-edit.
localparam integer L_CIN [0:8] = '{3,8,8,16,16,32,32,48,48};
localparam integer L_COUT [0:8] = '{8,8,16,16,32,32,48,48,48};
localparam integer L_KERNEL [0:8] = '{3,3,1,3,1,3,1,3,1};
localparam integer L_STRIDE [0:8] = '{2,2,1,2,1,2,1,1,1};
localparam integer L_PAD [0:8] = '{1,1,0,1,0,1,0,1,0};
localparam integer L_INPUT_HW [0:8] = '{96,48,24,24,12,12,6,6,6};
localparam integer L_OUTPUT_HW [0:8] = '{48,24,24,12,12,6,6,6,6};
localparam integer L_WEIGHT_OFFSET [0:8] = '{0,216,288,416,560,1072,1360,2896,3328};
localparam integer L_PARAMETER_OFFSET [0:8] = '{0,8,16,32,48,80,112,160,208};
localparam integer L_INPUT_PLANE [0:8] = '{9216,2304,576,576,144,144,36,36,36};
localparam integer L_OUTPUT_PLANE [0:8] = '{2304,576,576,144,144,36,36,36,36};
localparam integer L_KERNEL_AREA [0:8] = '{9,9,1,9,1,9,1,9,1};
localparam integer L_WEIGHT_STEP [0:8] = '{27,9,8,9,16,9,32,9,48};
localparam bit L_DEPTHWISE [0:8] = '{1'b0,1'b1,1'b0,1'b1,1'b0,1'b1,1'b0,1'b1,1'b0};
