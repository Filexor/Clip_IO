# Clip I/O
Clip I/O is extension of Stable diffusion Web UI.
This extension allows you to investigate embeddings/conditioning and feed conditioning to Stable Diffusion.  
## Clip Output
Clip Output allows you to export embeddings/conditioning whitch processed by Clip for investigation.  
### Options
#### Transpose matrix
Swap column (token axis) and row (dimension axis).   
#### Don't add bos / eos / pad tokens
In default settings, automatically append pad (padding) tokens to make chunk 75 tokens long then prepend bos (start of text) token and append eos (end of text) token.  
This option disables that behavior.
#### Ignore emphasis
This option makes multiplier of all tokens to 1.0 .  
Note that unlike disabling Settings -> Stable Diffusion -> Enable emphasis, emphasis syntax is still parsed.  
#### Bypass conditioning normalization
In default settings, mean average of emphasized conditioning will be adjusted to match mean average of pre-emphasized conditioning.  
This option disabled that behavior.  
#### Add token strings
Add token string for readability.  
Location of token string is affected by "Transpose matrix" option.  
#### Overwrite
Overwrite pre-existing file with same name as Output name.
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
### Syntax for "Directive" mode
**NOTE: Currently, because of syntax mess, "Directive" mode does not support Prompt editing and Alternating words.**
In addition of "Simple" mode syntax, "Directive" mode supportes inline directives.  
The syntax of inline directive is:
?`DirectiveName`(`DirectiveInner`) or
?`DirectiveName`_`DirectiveOrder`(`DirectiveInner`)
"DirectiveName" is name of directive such as "eval" or "exec" (case-insensitive).  
"DirectiveOrder" is order of directive.
Larger order makes processing directive later.
If directives with same order exists, these directives will be processed from left to right.
If "DirectiveOrder" is absent, it will be treated as order is 0.
#### Directives
##### eval
"eval" does component-wise python's eval to conditioning.
Local objects for eval are:
i: torch.Tensor : input conditioning
o: torch.Tensor : output conditioning
t: int : 0th dimension (token-wise) of index of input conditioning
d: int : 1st dimension (dimension-wise) of index of input conditioning
torch module and all objects in math module
##### exec
"exec" does component-wise python's exec.
Local objects for exec are:
i: torch.Tensor : input conditioning
o: torch.Tensor : output conditioning
t: int : 0th dimension (token-wise) of index of input conditioning
d: int : 1st dimension (dimension-wise) of index of input conditioning
torch module and all objects in math module