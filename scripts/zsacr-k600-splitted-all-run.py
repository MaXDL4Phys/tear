import os

# Define values we want to iterate over
archs=['ViT-B/16']#, 'ViT-B/32']
prompts =[  'only_label']#, 'description', 'decomposition', 'context']
models = ['simple'] # only simple in interesting for our work #, 'ensemble']
templates = ['False']#'True'] #, 'False']
conditionings = ['False']#'True'] #, 'False']
splits = [ 3] # 1,2, 3,]
# Iterate over the values and run the appropriate command
for arch in archs:
    print(f"Arch: {arch}")
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        for model in models:
            print(f"Model: {model}")
            for template in templates:
                print(f"Template: {template}")
                for conditioning in conditionings:
                    print(f"Conditioning: {conditioning}")
                    name = 'ViTb32' if arch == 'ViT-B/32' else 'ViTb16'
                    for split in splits:
                        print(f"Split: {split}")
                        command = f"""
                        CUDA_VISIBLE_DEVICES=0 \
                        HYDRA_FULL_ERROR=1 \
                        WANDB_DISABLE_SERVICE=True \
                        python -m src.eval  experiment="k600_splitted" \
                        model.decomposition.alpha=0.8 \
                        model.network.arch={arch} \
                        model.network.temperature=0.5 \
                        +experiment.data.split={split} \
                        logger.wandb.offline=False \
                        logger.wandb.tags=["tear_k600"] \
                        logger.wandb.project="TEzsAR-k600-{name}-{prompt}" \
                        logger.wandb.name={name}-{prompt}-{template}-{conditioning}-{split} \
                        trainer=gpu \
                        trainer.devices=1 \
                        data.num_workers=16 \
                        model.decomposition.use_templates={template} \
                        model.decomposition.prompts={prompt} \
                        model.decomposition.input_conditioning={conditioning} \
                        model.method={model} \
                        data.batch_size=16
                        """
                        os.system(command)

