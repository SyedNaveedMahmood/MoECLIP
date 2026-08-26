param(
    [string]$Conda = "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    [string]$DataRoot = "data\MulSenAD_official\MulSen_AD"
)

$ErrorActionPreference = "Stop"
$Stats = "data\MulSenAD_official\thermal_stats_final.json"
$RunA = "ckpt\mulsen_final_A_corrected_seed111"
$RunD = "ckpt\mulsen_final_D_v1_1_seed111"

# The statistics command reads only normal training images from final D_s.
& $Conda run --no-capture-output -n moeclip python `
    tools\compute_mulsen_thermal_stats.py `
    --data-root $DataRoot --protocol-stage final --output $Stats

$Common = @(
    "--dataset", "MulSenAD",
    "--data_root", $DataRoot,
    "--protocol_stage", "final",
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
    "--thermal_depth", "4",
    "--thermal_width", "256",
    "--region_context_dim", "256",
    "--region_attention_heads", "4",
    "--region_coordinate_bias", "1.0",
    "--region_coordinate_sigma", "0.75",
    "--num_context_experts", "4",
    "--modality_dropout", "0.2",
    "--thermal_aux_lambda", "0.0",
    "--align_loss_lambda", "0.0",
    "--adapter_norm_floor", "1.0",
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

# Use fresh output paths when reproducing: writers refuse to overwrite results.
& $Conda run --no-capture-output -n moeclip python train_mulsen.py @Common `
    --variant A --epochs 9 --output_dir $RunA

& $Conda run --no-capture-output -n moeclip python train_mulsen.py @Common `
    --variant D --epochs 3 --thermal_stats $Stats --use_global_context `
    --output_dir $RunD

# Each final model is evaluated once at its development-fixed epoch. Passing a
# checkpoint file instead of a directory prevents any final checkpoint scan.
& $Conda run --no-capture-output -n moeclip python evaluate_mulsen.py `
    --checkpoint "$RunA\mulsen_epoch_009.pth" --data_root $DataRoot `
    --output "$RunA\final_evaluation.json" --batch_size 1 --workers 4

& $Conda run --no-capture-output -n moeclip python evaluate_mulsen.py `
    --checkpoint "$RunD\mulsen_epoch_003.pth" --data_root $DataRoot `
    --output "$RunD\final_evaluation.json" --batch_size 1 --workers 4

& $Conda run --no-capture-output -n moeclip python `
    tools\inspect_mulsen_routing.py `
    --checkpoint "$RunD\mulsen_epoch_003.pth" --data_root $DataRoot `
    --output "$RunD\routing_audit_final.json" --batch_size 1 --workers 4
