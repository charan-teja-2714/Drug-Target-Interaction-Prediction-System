#!/bin/bash
# Git Repository Corruption Fix Script

echo "=== Git Repository Corruption Fix ==="
echo ""

# Step 1: Backup current work
echo "Step 1: Creating backup..."
cd ..
cp -r DTI_Prediction DTI_Prediction_backup
cd DTI_Prediction
echo "✓ Backup created at ../DTI_Prediction_backup"
echo ""

# Step 2: Remove corrupted pack file
echo "Step 2: Removing corrupted pack files..."
rm -f .git/objects/pack/pack-9635e49f9e6ba3e1fc7e39ed440b4d619da0aa43.*
echo "✓ Corrupted pack removed"
echo ""

# Step 3: Run git fsck
echo "Step 3: Running git fsck..."
git fsck --full
echo ""

# Step 4: Repack repository
echo "Step 4: Repacking repository..."
git repack -a -d
echo "✓ Repository repacked"
echo ""

# Step 5: Garbage collection
echo "Step 5: Running garbage collection..."
git gc --aggressive --prune=now
echo "✓ Garbage collection complete"
echo ""

echo "=== Fix Complete ==="
echo "Try: git push -u origin main"
