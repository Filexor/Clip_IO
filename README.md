# Clip I/O
Clip I/O is extension of Stable diffusion Web UI.
This extension allows you to investigate embeddings/conditioning and feed conditioning to Stable Diffusion.  
## Clip Output
Clip Output allows you to export embeddings/conditioning whitch processed by Clip for investigation.  
## Clip Input
Clip Input is, in spite of its name, bypasses Clip and allows you to feed your conditioning to Stable Diffusion model.  
This is useful not just for investigation, but using special conditioning which is impossible with using Clip.  
e.g. Combining two conditionings which exported with different Clip Skip settings.  
### How to use
1. Expand "Clip Input" accordion.
2. Enable "Enable" checkbox.
3. Select "Positive/Negative prompt mode".
Note: Clip Input uses Positive/Negative Prompt textbox. For syntax, see section below.
### Syntax for "Simple" mode
Each space-separated text represents file in /extensions/Clip_IO/conditioning/ .  
For filename without extension, if it does not exist, will searched with appending ".csv" or ".pt" in that order.  
If filename has space, you can enclose with single double-quotation or single single-quotation.  
You can include prompt which to be processed by Clip, by enclosing prompt with triple double-quotation or triple single-quotation.  
After gathering all conditionings, these conditionings will be concaterated.  