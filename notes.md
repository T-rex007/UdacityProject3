# Sagemaker Debugger 
##  Steps 
- Import amazon debugert 
- add hooks to train and test the model
- create jhooks and register model 
- configure debugger rules

```
import smdebug.pytorch as smd
hook.set_mode(smd.modes.TRAIN)
hook.set_mode(smd.modes.EVAL)
hook = smd.Hook.create_from_json_file()
hook.register_hook(model)
.... main

train(args, model, device, train_loader, optimizer, epoch, hook)
test(model, device, test_loader, hook)


from sagemaker.debugger import Rule, DebuggerHookConfig

rules = [
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.overtraining()),
    Rule.sagemaker(rule_configs.poor_weight_initialization()),
]

hook_config = DebuggerHookConfig(
    hook_parameters={"train.save_interval": "100", "eval.save_interval": "10"}
)

```

