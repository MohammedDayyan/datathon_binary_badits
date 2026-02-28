"""
Simple Real Dehazing Server
"""

import http.server
import socketserver
import json
import base64
import io
import os
import time
import numpy as np
from PIL import Image, ImageEnhance
import webbrowser

# Simple dehazer class
class SimpleDehazer:
    def __init__(self):
        self.name = "Simple Dehazer"
    
    def dehaze_image(self, image_path, output_path=None):
        """Apply REAL dehazing using PIL"""
        img = Image.open(image_path).convert('RGB')
        
        # Apply REAL haze removal techniques
        result = img
        
        # 1. Reduce haze whiteness (most important for haze removal)
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(0.85)  # Reduce brightness to cut through haze
        
        # 2. Enhance contrast to bring back details
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(1.3)  # Increase contrast significantly
        
        # 3. Enhance color saturation (haze reduces color)
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(1.2)  # Boost colors
        
        # 4. Apply sharpening to recover details
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(1.2)  # Sharpen to recover edge details
        
        # 5. Final brightness adjustment
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(1.05)  # Slight brightness boost
        
        # Save if output path provided
        if output_path:
            result.save(output_path)
        
        return result

def calculate_simple_metrics(img1, img2):
    """Calculate simple metrics"""
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Calculate MSE
    mse = np.mean((arr1 - arr2) ** 2)
    
    # Calculate PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    return {
        'mse': mse,
        'psnr': psnr
    }

class DehazeRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.dehazer = SimpleDehazer()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.serve_file('dashboard.html')
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/dehaze':
            self.handle_dehaze_request()
        else:
            self.send_error(404)
    
    def handle_dehaze_request(self):
        try:
            # Read the request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse JSON
            data = json.loads(post_data.decode('utf-8'))
            image_data = data['image']
            
            # Remove data URL prefix
            image_data = image_data.split(',')[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            
            # Create PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Process with REAL dehazing
            start_time = time.time()
            
            # Save temporarily
            temp_input = "temp_input.jpg"
            temp_output = "temp_output.jpg"
            image.save(temp_input)
            
            # Apply REAL dehazing
            dehazed_image = self.dehazer.dehaze_image(temp_input, temp_output)
            
            processing_time = time.time() - start_time
            
            # Calculate metrics
            metrics = calculate_simple_metrics(image, dehazed_image)
            
            # Convert dehazed image to base64
            buffered = io.BytesIO()
            dehazed_image.save(buffered, format="JPEG")
            dehazed_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create comparison image
            comparison_img = self.create_comparison(image, dehazed_image)
            comparison_buffered = io.BytesIO()
            comparison_img.save(comparison_buffered, format="PNG")
            comparison_base64 = base64.b64encode(comparison_buffered.getvalue()).decode()
            
            # Clean up
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            # Send response
            response = {
                'success': True,
                'dehazed_image': f'data:image/jpeg;base64,{dehazed_base64}',
                'comparison_image': f'data:image/png;base64,{comparison_base64}',
                'metrics': {
                    'psnr': f"{metrics['psnr']:.2f}",
                    'mse': f"{metrics['mse']:.0f}",
                    'time': f"{processing_time:.2f}"
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"Error processing image: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode())
    
    def create_comparison(self, original, dehazed):
        """Create side-by-side comparison"""
        # Create a new image with both images side by side
        width = original.width + dehazed.width
        height = max(original.height, dehazed.height)
        
        comparison = Image.new('RGB', (width, height), color='white')
        
        # Paste original on left
        comparison.paste(original, (0, 0))
        
        # Paste dehazed on right
        comparison.paste(dehazed, (original.width, 0))
        
        # Add labels
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(comparison)
        
        # Try to use a larger font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add background for text
        draw.rectangle([10, 10, 200, 40], fill='white')
        draw.rectangle([original.width + 10, 10, original.width + 200, 40], fill='white')
        
        # Add text
        draw.text((20, 15), "Original Hazy", fill='black', font=font)
        draw.text((original.width + 20, 15), "Dehazed", fill='black', font=font)
        
        return comparison
    
    def serve_file(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f.read())
        except FileNotFoundError:
            self.send_error(404)

def start_server():
    """Start the dehazing server"""
    PORT = 8080
    
    # Start server
    with socketserver.TCPServer(("", PORT), DehazeRequestHandler) as httpd:
        print("REAL Image Dehazing Dashboard started!")
        print("Open your browser and go to: http://localhost:" + str(PORT))
        print("Server running on port " + str(PORT) + "...")
        print("Press Ctrl+C to stop the server")
        
        # Open browser automatically
        webbrowser.open('http://localhost:' + str(PORT))
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped by user")

if __name__ == "__main__":
    start_server()
