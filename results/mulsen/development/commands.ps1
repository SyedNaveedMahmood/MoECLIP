param(
    [string]$Conda = "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    [string]$DataRoot = "data\MulSenAD_official\MulSen_AD"
)

$ErrorActionPreference = "Stop"
$Stats = "data\MulSenAD_official\thermal_stats_development.json"
$RunA = "ckpt\mulsen_dev_A_corrected_seed111"
$RunD = "ckpt\mulsen_dev_D_v1_1_seed111"

$Common = @(
    "--dataset", "MulSenAD",
    "--data_root", $DataRoot,
    "--protocol_stage", "development",
    "--model_name", "ViT-L-14-336",
    "--img_size", "518",
    "--moe_r", "8",
    "--moe_lora_alpha", "16",
    "--moe_num_experts", "4",
    "--moe_top_k", "2",
    "--moe_layers", "5,11,17,23",
    "--router_init", "normal",
    "--image_adapt_weight", "0.1",
    "--seg_proj_sharing_strategy", "shared",
    "--num_context_experts", "4",
    "--modality_dropout", "0.2",
    "--align_loss_lambda", "0.0",
    "--adapter_norm_floor", "1.0",
    "--epochs", "20",
    "--batch_size", "1",
    "--workers", "4",
    "--lr", "5e-5",
    "--weight_decay", "0.0",
    "--lr_milestones", "12,16",
    "--lr_gamma", "0.1",
    "--balance_loss_lambda", "0.01",
    "--etf_loss_lambda", "0.01",
    "--amp_init_scale", "1024",
    "--seed", "111"
)

# Completed commands are retained for provenance. Each writer refuses to
# overwrite an existing output, so choose new directories to reproduce them.
& $Conda run --no-capture-output -n moeclip python train_mulsen.py @Common `
    --variant A --output_dir $RunA

& $Conda run --no-capture-output -n moeclip python evaluate_mulsen.py `
    --checkpoint_dir $RunA --data_root $DataRoot `
    --output "$RunA\development_evaluation.json" --batch_size 1 --workers 4

& $Conda run --no-capture-output -n moeclip python train_mulsen.py @Common `
    --variant D --thermal_stats $Stats --use_global_context --output_dir $RunD

& $Conda run --no-capture-output -n moeclip python evaluate_mulsen.py `
    --checkpoint_dir $RunD --data_root $DataRoot `
    --output "$RunD\development_evaluation.json" --batch_size 1 --workers 4

& $Conda run --no-capture-output -n moeclip python `
    tools\inspect_mulsen_routing.py `
    --checkpoint "$RunD\mulsen_epoch_003.pth" --data_root $DataRoot `
    --output "$RunD\routing_audit_selected.json" --batch_size 1 --workers 0
