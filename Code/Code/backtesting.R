base_path <- file.path("C:", "Users", "XiaoA1", "OneDrive - Moody's",
                      "COE - Credit Analytics-SQ - Unified ST Framework - Documents",
                      "Sovereign Macro", "0 - Reference", "GCorr Sovereign",
                      "Backtesting_for_Angela")
setwd(base_path) 

dyn.load("Code/MAAnalyticalMVModel.dll")
is.loaded("RStressedEL")
source("Code/functions.R") 

acc_key <- "FF91D2DD-FB35-4664-BDA5-E165A718941D"
enc_key <- "3F2579A0-99EF-487C-B0D9-0ECA49DFC46E"

aws_key = aws.signature::use_credentials(profile = "default") 