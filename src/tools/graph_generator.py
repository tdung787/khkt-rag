"""
src/tools/graph_generator.py

Tool để vẽ đồ thị hàm số bằng AI + Local Python Execution
Cost: ~$0.0003/request (rẻ nhất!)
"""

import os
import subprocess
import tempfile
import hashlib
import time
import re
from pathlib import Path
from typing import Dict, Optional
from openai import OpenAI

# Config
GRAPH_OUTPUT_DIR = "static/graphs"
EXECUTION_TIMEOUT = 10  # seconds


class GraphGenerator:
    """Generate math function graphs using AI + local Python execution"""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        
        # Ensure output directory exists
        Path(GRAPH_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    def generate_code(self, equation: str, x_min: float = -10, x_max: float = 10) -> str:
        """
        Use AI to generate Python code for plotting
        
        Cost: ~$0.0002/request
        
        Args:
            equation: Math equation (e.g., "x**2 + 2*x - 3")
            x_min: Minimum x value
            x_max: Maximum x value
            
        Returns:
            Python code as string
        """
        
        prompt = f"""Generate Python code to plot the function: y = {equation}

Requirements:
- Use matplotlib and numpy ONLY (no other imports)
- x range: [{x_min}, {x_max}] with 1000 points
- Vietnamese labels (x, y, title)
- Enable grid with alpha=0.3
- Add horizontal and vertical axis lines at zero
- Figure size: (10, 6)
- DPI: 100
- Save to 'OUTPUT_FILE_PATH' (will be replaced)
- Close plot after saving
- Handle mathematical functions: sin, cos, tan, log, exp, sqrt, abs
- Use numpy for math functions (np.sin, np.cos, etc.)

Output ONLY valid Python code, no explanations or markdown.

Example structure:
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace({x_min}, {x_max}, 1000)
y = {equation}

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2, color='blue')
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title(f'Đồ thị: y = {equation}', fontsize=14)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.savefig('OUTPUT_FILE_PATH', dpi=100, bbox_inches='tight')
plt.close()
```
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Python code generator. Output only clean, executable Python code. No explanations, no markdown blocks."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=600
        )
        
        code = response.choices[0].message.content.strip()
        
        # Extract code from markdown if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        return code
    
    def execute_code(self, code: str, output_filepath: str) -> Dict:
        """
        Execute Python code locally and save graph to file
        
        Cost: $0 (runs locally)
        
        Args:
            code: Python code to execute
            output_filepath: ABSOLUTE path where graph should be saved
            
        Returns:
            Dict with success status and file info
        """
        
        # Convert to absolute path
        output_path = Path(output_filepath).resolve()
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Replace placeholder with actual output path (use absolute path)
        code = code.replace('OUTPUT_FILE_PATH', str(output_path))
        code = code.replace("'output.png'", f"'{output_path}'")
        code = code.replace('"output.png"', f'"{output_path}"')
        
        # Create temp directory for execution
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Write code to temp file
            code_file = tmpdir_path / "graph_script.py"
            code_file.write_text(code, encoding='utf-8')
            
            try:
                # Execute code
                result = subprocess.run(
                    ["python", str(code_file)],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=EXECUTION_TIMEOUT
                )
                
                # Check execution result
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Execution failed: {result.stderr[:500]}"
                    }
                
                # Check if output file was created
                if not output_path.exists():
                    return {
                        "success": False,
                        "error": "Graph file was not generated"
                    }
                
                # Get file info
                file_size = output_path.stat().st_size
                
                return {
                    "success": True,
                    "file_path": str(output_path),
                    "file_size": file_size
                }
                
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"Execution timeout ({EXECUTION_TIMEOUT}s)"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Execution error: {str(e)}"
                }
    
    def generate_graph(
        self, 
        equation: str, 
        x_min: float = -10, 
        x_max: float = 10
    ) -> Dict:
        """
        Main method: Generate code + Execute + Save graph
        
        Total cost: ~$0.0003/request
        
        Args:
            equation: Math equation
            x_min: Min x value
            x_max: Max x value
            
        Returns:
            Dict with result info
        """
        
        print(f"\n📊 Generating graph for: y = {equation}")
        print(f"   📏 Range: [{x_min}, {x_max}]")
        
        # Generate unique filename with ABSOLUTE PATH
        unique_id = hashlib.md5(
            f"{equation}{x_min}{x_max}{time.time()}".encode()
        ).hexdigest()[:12]
        
        # Use absolute path
        output_filepath = Path(GRAPH_OUTPUT_DIR).resolve() / f"graph_{unique_id}.png"
        
        # Ensure directory exists
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"   💾 Output: {output_filepath}")
        
        # Step 1: Generate code using AI
        print("   🤖 AI generating Python code...")
        try:
            code = self.generate_code(equation, x_min, x_max)
            print(f"   ✓ Generated {len(code)} characters of code")
        except Exception as e:
            return {
                "success": False,
                "error": f"Code generation failed: {str(e)}",
                "equation": equation
            }
        
        # Step 2: Execute code locally
        print("   🐍 Executing Python code...")
        result = self.execute_code(code, str(output_filepath))  # ← Pass absolute path
        
        if result["success"]:
            print(f"   ✓ Graph saved to: {result['file_path']}")
            print(f"   📦 File size: {result['file_size']/1024:.1f}KB")
            
            # Add metadata
            result["equation"] = equation
            result["x_range"] = [x_min, x_max]
            result["generated_code"] = code  # For debugging
        else:
            print(f"   ✗ Failed: {result['error']}")
        
        return result


def extract_equation_from_query(query: str, openai_client: OpenAI) -> Optional[str]:
    """
    Extract mathematical equation from natural language query
    
    Args:
        query: User query in Vietnamese
        openai_client: OpenAI client
        
    Returns:
        Equation in Python syntax or None
    """
    
    # Try simple regex patterns first
    patterns = [
        r'y\s*=\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\.\s]+)',   # y = e^(-0.1*x^2) * cos(3*x) + 0.5*x
        r'vẽ\s+(?:đồ\s+thị\s+)?(.+?)(?:\s+từ|\s*$)',
        r'đồ\s+thị\s+(?:hàm\s+)?(.+?)(?:\s+từ|\s*$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            eq = match.group(1).strip()
            
            # Clean up common notation
            eq = eq.replace('^', '**')  # x^2 → x**2
            eq = eq.replace('×', '*')   # × → *
            eq = eq.replace('÷', '/')   # ÷ → /
            
            # Remove trailing punctuation
            eq = eq.rstrip('.,;:!?')
            
            return eq
    
    # Fallback: Use AI to extract
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Extract the mathematical equation from this query: "{query}"

Return ONLY the equation in Python syntax (e.g., x**2, sin(x), np.cos(x), etc.).
Use numpy functions: np.sin, np.cos, np.tan, np.log, np.exp, np.sqrt, np.abs
No explanations, just the equation.

Examples:
"vẽ đồ thị y = x bình phương" → "x**2"
"vẽ hàm sin x" → "np.sin(x)"
"đồ thị y = 2x + 3" → "2*x + 3"
"""
            }],
            temperature=0,
            max_tokens=100
        )
        
        equation = response.choices[0].message.content.strip()
        
        # Remove quotes if present
        equation = equation.strip('"\'')
        
        return equation if equation else None
        
    except Exception as e:
        print(f"⚠️  Error extracting equation: {e}")
        return None


def extract_range_from_query(query: str) -> tuple:
    """
    Extract x range from query
    
    Returns:
        (x_min, x_max) or (-10, 10) as default
    """
    
    # Pattern: "từ -5 đến 5", "from -10 to 10"
    pattern = r'từ\s+(-?\d+)\s+đến\s+(-?\d+)|from\s+(-?\d+)\s+to\s+(-?\d+)'
    match = re.search(pattern, query, re.IGNORECASE)
    
    if match:
        # Get first non-None group pair
        groups = match.groups()
        x_min = float(groups[0] or groups[2])
        x_max = float(groups[1] or groups[3])
        return (x_min, x_max)
    
    return (-10, 10)  # Default range