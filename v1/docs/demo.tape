# SVA Toolkit Demo Script
# Record with: vhs docs/demo.tape

Output docs/demo.mp4

Set Shell bash
Set FontSize 20
Set Width 1400
Set Height 1000
Set LetterSpacing 1
Set LineHeight 1.4

# Overview: Toolkit Components
Copy "# ============================================================================"
Paste
Enter
Copy "# SVA TOOLKIT - Complete Suite for Assertion Generation & Verification"
Paste
Enter
Copy "# ============================================================================"
Paste
Enter
Sleep 1s
Copy "#"
Paste
Enter
Copy "# -----------------------------------------------------------------------------"
Paste
Enter
Copy "# Tool              | Description"
Paste
Enter
Copy "# ------------------|----------------------------------------------------------"
Paste
Enter
Copy "# sva-gen           | Generate syntactically-valid SVA properties"
Paste
Enter
Copy "# sva-implication   | Verify equivalence and implication relationships"
Paste
Enter
Copy "# svad-translator   | Translate between SVA code and natural language"
Paste
Enter
Copy "# sva-benchmark     | Benchmark LLM performance on SVA tasks"
Paste
Enter
Copy "# sva-dataset       | Build training datasets with SVA-SVAD pairs"
Paste
Enter
Copy "# sva-cot           | Generate chain-of-thought reasoning for synthesis"
Paste
Enter
Copy "# sva-ast           | Parse SVA code into abstract syntax trees"
Paste
Enter
Copy "# -----------------------------------------------------------------------------"
Paste
Enter
Copy "#"
Paste
Enter
Sleep 6s
Copy "# Let's explore the sva-gen and sva-implication-checker with live demonstrations..."
Paste
Enter
Sleep 2s
Ctrl+L

# Section 1: List Available Signal Presets
Type "# 1. Explore available signal presets"
Enter
Sleep 1s
Type "sva-gen list-presets"
Enter
Sleep 3s
Type "# Four built-in presets for different protocols (default, handshake, fifo, axi)"
Enter
Sleep 3s
Ctrl+L

# Section 2: Generate SVA Properties
Type "# 2. Generate SVA properties with handshake signals"
Enter
Sleep 1s
Type "sva-gen generate -n 5 --preset handshake --max-depth 2"
Enter
Sleep 6s
Type "# Generated 5 assertions - All syntax validated with Verible"
Enter
Sleep 3s
Ctrl+L

# Section 3: Generate with custom signals
Type "# 3. Generate with custom signals and save to file"
Enter
Sleep 1s
Type 'sva-gen generate -n 3 -s req -s ack -s valid -s ready -o /tmp/demo.sv'
Enter
Sleep 3s
Type "cat /tmp/demo.sv | head -20"
Enter
Sleep 5s
Type "# Custom signals: req, ack, valid, ready - Output saved to /tmp/demo.sv"
Enter
Sleep 3s
Ctrl+L

# Section 4: Validate SVA syntax
Type "# 4. Validate SVA syntax"
Enter
Sleep 1s
Type 'sva-gen validate "module m; assert property (@(posedge clk) req |-> ##1 ack); endmodule"'
Enter
Sleep 4s
Type "# Syntax validation powered by Verible - Catches errors before saving to dataset"
Enter
Sleep 3s
Ctrl+L

# Section 5: Check equivalence - Real example
Type "# 5. Check equivalence: Two ways to express read acknowledgment"
Enter
Sleep 1s
Type 'sva-implication equivalent --sva1 "(!o_wb_we && i_wb_ack) |-> (read_ack == (i_wb_ack & ~o_wb_we))" --sva2 "(o_wb_we == 0 && i_wb_ack == 1) |-> (read_ack == (~o_wb_we & i_wb_ack))"'
Enter
Sleep 6s
Type "# Result: EQUIVALENT - Both assertions express the same property"
Enter
Sleep 2s
Type "# SVA1 uses logical NOT (!), SVA2 uses comparison (== 0)"
Enter
Sleep 3s
Ctrl+L

# Section 6: Check implication - Stronger condition
Type "# 6. Check implication: Which reset condition is stronger?"
Enter
Sleep 1s
Type 'sva-implication check -a "grst == 1 |-> ##1 bra_ex == 2'"'"'h0" -c "$rose(grst) |=> (bra_ex == 2'"'"'h0)"'
Enter
Sleep 6s
Type "# Result: Antecedent IMPLIES consequent - Antecedent is STRONGER"
Enter
Sleep 2s
Type "# Antecedent (grst==1) triggers EVERY cycle when high"
Enter
Sleep 2s
Type "# Consequent ($rose) triggers only on RISING EDGE - Less restrictive"
Enter
Sleep 3s
Ctrl+L

# Section 7: Analyze relationship
Type "# 7. Analyze bidirectional relationship"
Enter
Sleep 1s
Type 'sva-implication relationship --sva1 "req |-> ##[1:3] ack" --sva2 "req |=> ack"'
Enter
Sleep 5s
Type "# Checks BOTH directions: SVA1 implies SVA2? and SVA2 implies SVA1?"
Enter
Sleep 2s
Type "# SVA2 (##1) is more restrictive than SVA1 (##[1:3])"
Enter
Sleep 3s
Ctrl+L

# Section 8: Coverage analysis of generated dataset
Type "# 8. Generate dataset and analyze construct coverage"
Enter
Sleep 1s
Type "sva-gen generate -n 100 --mode stratified --samples-per-construct 3 -j -o /tmp/demo_dataset.json"
Enter
Sleep 3s
Type "# Stratified mode ensures balanced coverage across constructs"
Enter
Sleep 2s
Type "sva-gen coverage /tmp/demo_dataset.json --top-n 5"
Enter
Sleep 7s
Type "# Coverage analysis: Tracks 48 SVA constructs across 7 categories"
Enter
Sleep 2s
Type "# Color-coded tables show top 5 constructs per category"
Enter
Sleep 2s
Type "# Essential for dataset quality assurance"
Enter
Sleep 3s
Ctrl+L

# Outro
Type "#"
Enter
Type "# SVA Toolkit: Complete Assertion Generation & Verification"
Enter
Type "#"
Enter
Sleep 1s
Type "sva-gen --help"
Enter
Sleep 6s
Type "# Generate syntactically-valid SVA properties with Verible validation"
Enter
Sleep 2s
Ctrl+L
Type "sva-implication --help"
Enter
Sleep 6s
Type "# Formally verify equivalence and implication relationships between SVAs"
Enter
Sleep 2s
Ctrl+L
Type "svad-translator --help"
Enter
Sleep 6s
Type "# Translate between SVA code and golden natural language descriptions (SVAD), using the same type system as the sva-gen"
Enter
Sleep 2s
Ctrl+L
Type "sva-benchmark --help"
Enter
Sleep 6s
Type "# Benchmark LLM performance on SVA generation and translation tasks"
Enter
Sleep 2s
Ctrl+L
Type "sva-dataset --help"
Enter
Sleep 6s
Type "# Build training datasets with (LLM-powered) SVA-SVAD pairs for LLM fine-tuning"
Enter
Sleep 2s
Ctrl+L
Type "sva-cot --help"
Enter
Sleep 6s
Type "# Generate (LLM-powered) chain-of-thought reasoning for SVA synthesis"
Enter
Sleep 2s
Ctrl+L
Type "sva-ast --help"
Enter
Sleep 6s
Type "# Parse SVA code into abstract syntax trees for analysis"
Enter
Sleep 2s
Ctrl+L
Type "#"
Enter
Copy "# ============================================================"
Paste
Enter
Type "# TRY SVA-TOOLKIT NOW!"
Enter
Copy "# ============================================================"
Paste
Enter
Sleep 3s
