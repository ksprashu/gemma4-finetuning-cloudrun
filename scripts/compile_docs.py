#!/usr/bin/env python3
"""
Gemma 4 Fine-Tuning & Serving Encyclopedia - Documentation Compiler
This script compiles modular chapters from knowledgebase/src/chapters/ 
and wraps them inside knowledgebase/src/shell_template.html 
to build the highly optimized, offline-friendly single-page production index.html.
"""

import os
import sys
import time

def main():
    start_time = time.time()
    
    # Establish base and source paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kb_dir = os.path.join(base_dir, "knowledgebase")
    src_dir = os.path.join(kb_dir, "src")
    chapters_dir = os.path.join(src_dir, "chapters")
    shell_template_path = os.path.join(src_dir, "shell_template.html")
    output_index_path = os.path.join(kb_dir, "index.html")
    
    print("====================================================")
    print("  Gemma 4 Encyclopedia Compiler - Building Production")
    print("====================================================")
    
    # Check if files exist
    if not os.path.exists(shell_template_path):
        print(f"Error: Shell template not found at {shell_template_path}", file=sys.stderr)
        sys.exit(1)
        
    # Read the layout template
    with open(shell_template_path, 'r', encoding='utf-8') as f:
        shell_content = f.read()
        
    # Sequence of chapters to compile
    chapters_order = [
        "introduction",
        "chapter1",
        "chapter2",
        "chapter3",
        "chapter4",
        "chapter5",
        "chapter6",
        "chapter7",
        "chapter8",
        "chapter9"
    ]
    
    compiled_chapters = []
    
    for ch in chapters_order:
        ch_file_path = os.path.join(chapters_dir, f"{ch}.html")
        if not os.path.exists(ch_file_path):
            print(f"Warning: Chapter file '{ch}.html' not found in {chapters_dir}.", file=sys.stderr)
            continue
            
        print(f" -> Reading {ch}.html...")
        with open(ch_file_path, 'r', encoding='utf-8') as cf:
            ch_content = cf.read()
            
        compiled_chapters.append(ch_content)
        
    # Standardize separator spacing
    chapters_combined = "\n\n" + "\n\n".join(compiled_chapters) + "\n\n"
    
    # Replace placeholder token in template
    placeholder = "<!-- {{CHAPTERS_CONTENT}} -->"
    if placeholder not in shell_content:
        print(f"Error: Placeholder token '{placeholder}' not found in shell_template.html", file=sys.stderr)
        sys.exit(1)
        
    compiled_html = shell_content.replace(placeholder, chapters_combined)
    
    # Save the output production file
    with open(output_index_path, 'w', encoding='utf-8') as out_f:
        out_f.write(compiled_html)
        
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    file_size_kb = os.path.getsize(output_index_path) / 1024
    
    print("----------------------------------------------------")
    print(f"✓ Compilation successful!")
    print(f"✓ Output file: {output_index_path}")
    print(f"✓ Total size: {file_size_kb:.2f} KB")
    print(f"✓ Time elapsed: {elapsed_ms:.2f} ms")
    print("====================================================")

if __name__ == "__main__":
    main()
