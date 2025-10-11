# Prompt Simplification Summary

## Problem Statement

The original prompt was too complex and contained conflicting instructions that caused the model to:
1. Modify parts of the person's photo that should remain unchanged (face, hair, body, background)
2. Alter clothing details from the reference image
3. Change the person's orientation/viewing angle

## Root Cause

The prompt tried to do too much at once:
- Long verbose instructions about what to preserve
- Complex orientation requirements embedded in the middle
- Conflicting messages: "keep everything the same" vs "replace clothing"
- Over 400+ characters of instructions that confused the model

## Solution: Simplified Prompt Structure

### Before (Complex)
```
This is a person wearing clothes. Keep the person's identity, facial expression, hair, 
body shape, skin tone, and background exactly the same. **CRITICAL ORIENTATION REQUIREMENT: 
The person is facing FORWARD with their face clearly visible to the camera. Maintain this 
FRONT-FACING orientation exactly - keep the face visible and forward-looking.** In the 
masked region, replace the person's current clothing with this new garment: [description]. 
The new clothing should fit naturally on the person's body, following their exact posture, 
limb positioning, and proportions. Respect the original lighting, shadows, and scene context 
so the new outfit blends seamlessly into the photo. Preserve every critical visual detail, 
surface quality, and embellishment of the new garment: [fine_details]
```
**Length:** ~550+ characters

### After (Simplified)
```
Replace only the clothing in the masked region with: [description]. 
Keep front-facing view (face visible). 
Preserve clothing details: [fine_details]
```
**Length:** ~250 characters (55% reduction)

## Key Changes

### 1. Main Prompt Simplification
- **Before:** Long explanation of what to preserve + what to change
- **After:** Direct instruction: "Replace only the clothing in the masked region"
- **Benefit:** Clear focus on the task, no conflicting instructions

### 2. Orientation Instructions
- **Before:** 2-3 sentences with emphasis and repetition
- **After:** Single short sentence (e.g., "Keep front-facing view")
- **Benefit:** Clear, memorable, unambiguous

### 3. Enhanced Negative Prompt
Now explicitly lists ALL unwanted changes:
```
changing face, changing hair, changing skin tone, changing body shape, changing pose, 
changing background, changing person identity, modifying non-clothing areas, 
altering facial features, different hairstyle, different hair color, 
rotating person, turning around, changing viewing angle, flipped orientation, 
different camera angle, modified perspective, changed body position, 
altered arm position, altered leg position, different stance, 
extra limbs, missing limbs, distorted anatomy, unnatural proportions, 
modifying garment design, changing clothing colors, altering clothing patterns, 
changing fabric texture shown in reference, simplified clothing details
```

## Expected Benefits

1. **Person Photo Preservation:** The model receives clear negative examples of what NOT to change
2. **Clothing Detail Preservation:** Negative prompt explicitly prevents modifying garment design/colors/patterns
3. **Orientation Preservation:** Short, clear instruction is easier for the model to follow
4. **Reduced Confusion:** No conflicting "preserve everything while changing something" messages

## Testing

Run the test script to see the new prompts:
```bash
.venv/bin/python test_simplified_prompt.py
```

## Implementation Details

Modified files:
- [`src/azure_bfl_compare/tasks/outfit_transfer.py`](../src/azure_bfl_compare/tasks/outfit_transfer.py)
  - [`_build_flux_prompt()`](../src/azure_bfl_compare/tasks/outfit_transfer.py:1023) - Simplified prompt structure
  - [`_get_orientation_instruction()`](../src/azure_bfl_compare/tasks/outfit_transfer.py:355) - Shortened orientation messages

## Comparison Table

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| Prompt Length | ~550 chars | ~250 chars | 55% shorter |
| Orientation Instruction | 2-3 sentences | 1 sentence | 70% shorter |
| Clarity | Mixed messages | Clear directive | ✓ Better |
| Negative Prompt | Generic | Explicit list | ✓ Comprehensive |
| Focus | Scattered | Laser-focused | ✓ Improved |

## Philosophy

**KISS Principle (Keep It Simple, Stupid):**
- Tell the model exactly what to DO
- Use negative prompts to list what NOT to do
- Remove verbose explanations and repetition
- Trust that shorter, clearer instructions work better than long complex ones