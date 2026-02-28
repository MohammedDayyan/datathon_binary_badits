"""
PROFESSIONAL Image Dehazer
Uses proper atmospheric haze removal algorithms while preserving image quality
"""

import http.server
import socketserver
import json
import base64
import io
import os
import time
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import webbrowser

class ProfessionalDehazer:
    def __init__(self):
        self.name = "Professional Dehazer"
    
    def dehaze_image(self, image_input, output_path=None):
        """Apply professional atmospheric haze removal"""
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input.convert('RGB')
        
        print(f"Professional dehazing: {img.size}")
        
        # Convert to numpy for processing
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Apply professional atmospheric haze removal
        result_array = self.atmospheric_dehazing(img_array)
        
        # Convert back to PIL
        result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
        result = Image.fromarray(result_array)
        
        # Apply subtle enhancements to restore quality
        result = self.restore_image_quality(result)
        
        print(f"Professional dehazing completed!")
        
        # Save if output path provided
        if output_path:
            result.save(output_path, quality=98)
        
        return result
    
    def atmospheric_dehazing(self, img_array):
        """Professional atmospheric haze removal algorithm"""
        # Dark Channel Prior - fundamental dehazing technique
        dark_channel = self.compute_dark_channel(img_array)
        
        # Estimate atmospheric light
        atmospheric_light = self.estimate_atmospheric_light(img_array, dark_channel)
        
        # Estimate transmission map
        transmission = self.estimate_transmission(img_array, dark_channel, atmospheric_light)
        
        # Refine transmission map
        transmission = self.refine_transmission(transmission, img_array)
        
        # Recover scene radiance (dehazed image)
        result = self.recover_scene_radiance(img_array, transmission, atmospheric_light)
        
        return result
    
    def compute_dark_channel(self, img_array, patch_size=15):
        """Compute dark channel prior"""
        import cv2
        dark_channel = np.min(img_array, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(dark_channel, kernel)
        return dark_channel
    
    def estimate_atmospheric_light(self, img_array, dark_channel, percentile=0.001):
        """Estimate atmospheric light from brightest pixels in dark channel"""
        h, w = dark_channel.shape
        num_pixels = max(1, int(h * w * percentile))
        
        # Get top brightest pixels in dark channel
        flat_dark = dark_channel.flatten()
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        
        # Get corresponding pixels in original image
        bright_pixels = img_array.reshape(-1, 3)[indices]
        
        # Return the brightest pixel as atmospheric light
        atmospheric_light = np.max(bright_pixels, axis=0)
        
        return atmospheric_light
    
    def estimate_transmission(self, img_array, dark_channel, atmospheric_light, omega=0.85):
        """Estimate transmission map"""
        # Normalize image by atmospheric light
        normalized_img = img_array / atmospheric_light
        
        # Compute transmission using dark channel
        transmission = 1 - omega * self.compute_dark_channel(normalized_img)
        
        # Ensure minimum transmission to avoid noise amplification and color shifting in dense haze
        transmission = np.clip(transmission, 0.2, 1.0)
        
        return transmission
    
    def refine_transmission(self, transmission, img_array):
        """Refine transmission map using guided filter-like approach"""
        import cv2
        gray = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # Guided filter parameters
        r = 60
        eps = 0.0001
        
        I = gray
        p = transmission
        
        mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
        
        refined = mean_a * I + mean_b
        
        return refined
    
    def recover_scene_radiance(self, img_array, transmission, atmospheric_light):
        """Recover the scene radiance (dehazed image)"""
        # Add small epsilon to avoid division by zero
        epsilon = 0.001
        
        # Recover scene radiance using atmospheric scattering model
        transmission_3d = np.stack([transmission] * 3, axis=2)
        
        # J(x) = (I(x) - A) / max(t(x), t0) + A
        scene_radiance = (img_array - atmospheric_light) / np.maximum(transmission_3d, epsilon) + atmospheric_light
        
        # Clip to valid range
        scene_radiance = np.clip(scene_radiance, 0, 1)
        
        return scene_radiance
    
    def restore_image_quality(self, img):
        """Restore and enhance image quality after dehazing"""
        # Very subtle contrast enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.02)
        
        # Very subtle color enhancement
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.02)
        
        # Very subtle sharpening
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.01)
        
        # Remove UnsharpMask completely to prevent amplifying block noise and chromatic aberration in the sky
        
        return img

def calculate_professional_metrics(img1, img2):
    """Calculate professional image quality metrics"""
    from skimage.metrics import structural_similarity as ssim
    import cv2
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Calculate MSE
    mse = np.mean((arr1 - arr2) ** 2)
    
    # Calculate PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Calculate haze reduction metrics
    brightness_orig = np.mean(arr1)
    brightness_dehazed = np.mean(arr2)
    haze_reduction = max(0, brightness_orig - brightness_dehazed)
    
    # Calculate contrast improvement
    contrast_orig = np.std(arr1)
    contrast_dehazed = np.std(arr2)
    contrast_improvement = ((contrast_dehazed - contrast_orig) / contrast_orig) * 100
    
    # Calculate detail preservation (edge preservation)
    from scipy import ndimage
    edges_orig = ndimage.sobel(arr1.mean(axis=2))
    edges_dehazed = ndimage.sobel(arr2.mean(axis=2))
    detail_preservation = np.corrcoef(edges_orig.flatten(), edges_dehazed.flatten())[0,1]
    
    # Calculate SSIM
    gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)
    ssim_val = ssim(gray1, gray2, data_range=255)
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim_val,
        'haze_reduction': haze_reduction,
        'contrast_improvement': contrast_improvement,
        'detail_preservation': detail_preservation
    }

class ProfessionalDehazeRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.dehazer = ProfessionalDehazer()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.serve_file('professional_dashboard.html')
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/dehaze':
            self.handle_dehaze_request()
        else:
            self.send_error(404)
    
    def handle_dehaze_request(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            data = json.loads(post_data.decode('utf-8'))
            image_data = data['image']
            
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            print(f"Professional processing: {image.size}")
            
            start_time = time.time()
            
            dehazed_image = self.dehazer.dehaze_image(image)
            
            processing_time = time.time() - start_time
            
            metrics = calculate_professional_metrics(image, dehazed_image)
            
            buffered = io.BytesIO()
            dehazed_image.save(buffered, format="PNG")
            dehazed_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            comparison_img = self.create_professional_comparison(image, dehazed_image, metrics)
            comparison_buffered = io.BytesIO()
            comparison_img.save(comparison_buffered, format="PNG")
            comparison_base64 = base64.b64encode(comparison_buffered.getvalue()).decode()
            
            response = {
                'success': True,
                'dehazed_image': f'data:image/png;base64,{dehazed_base64}',
                'comparison_image': f'data:image/png;base64,{comparison_base64}',
                'metrics': {
                    'psnr': f"{metrics['psnr']:.2f}",
                    'ssim': f"{metrics['ssim']:.3f}",
                    'mse': f"{metrics['mse']:.0f}",
                    'time': f"{processing_time:.2f}",
                    'haze_reduction': f"{metrics['haze_reduction']:.1f}",
                    'contrast_improvement': f"{metrics['contrast_improvement']:.1f}%",
                    'detail_preservation': f"{metrics['detail_preservation']:.3f}"
                }
            }
            
            print(f"Professional dehazing completed in {processing_time:.2f}s")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"Error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode())
    
    def create_professional_comparison(self, original, dehazed, metrics):
        """Create professional comparison with metrics"""
        width = original.width + dehazed.width + 150
        height = max(original.height, dehazed.height) + 150
        
        comparison = Image.new('RGB', (width, height), color='white')
        
        comparison.paste(original, (20, 100))
        comparison.paste(dehazed, (original.width + 130, 100))
        
        draw = ImageDraw.Draw(comparison)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 28)
            label_font = ImageFont.truetype("arial.ttf", 20)
            metric_font = ImageFont.truetype("arial.ttf", 16)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            metric_font = ImageFont.load_default()
        
        # Title
        draw.text((width//2 - 200, 20), "Professional Atmospheric Dehazing", fill='#2196F3', font=title_font)
        
        # Labels
        draw.rectangle([20, 60, 250, 85], fill='#e3f2fd')
        draw.text((25, 62), "Original Hazy Image", fill='black', font=label_font)
        
        draw.rectangle([original.width + 130, 60, original.width + 360, 85], fill='#e8f5e8')
        draw.text((original.width + 135, 62), "Professional Dehazed", fill='black', font=label_font)
        
        # Metrics
        y_pos = original.height + 120
        draw.text((20, y_pos), "Professional Results:", fill='#2196F3', font=label_font)
        draw.text((20, y_pos + 25), f"Haze Reduction: {metrics['haze_reduction']:.1f}", fill='#666', font=metric_font)
        draw.text((20, y_pos + 45), f"Contrast Improvement: {metrics['contrast_improvement']:.1f}%", fill='#666', font=metric_font)
        draw.text((20, y_pos + 65), f"Detail Preservation: {metrics['detail_preservation']:.3f}", fill='#666', font=metric_font)
        draw.text((20, y_pos + 85), "Atmospheric haze removal applied", fill='#666', font=metric_font)
        draw.text((20, y_pos + 105), "Image quality and details preserved", fill='#666', font=metric_font)
        
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

def create_professional_dashboard():
    """Create the professional dashboard HTML"""
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Atmospheric Dehazing Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f8fafc;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --card-bg: #ffffff;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --border: #e2e8f0;
            --success: #10b981;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }

        .container {
            width: 100%;
            max-width: 900px;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-bottom: 8px;
        }

        .header p {
            color: var(--text-muted);
            font-size: 1rem;
            font-weight: 300;
        }

        .card {
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04);
            border: 1px solid var(--border);
            padding: 40px;
            margin-bottom: 24px;
        }

        .upload-area {
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 48px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: #f8fafc;
        }

        .upload-area:hover {
            border-color: var(--accent);
            background: #eff6ff;
        }

        .upload-area svg {
            width: 48px;
            height: 48px;
            color: #94a3b8;
            margin-bottom: 16px;
        }

        .upload-text {
            font-weight: 500;
            color: var(--text-main);
            margin-bottom: 4px;
        }

        .upload-hint {
            font-size: 0.875rem;
            color: var(--text-muted);
        }

        .btn {
            background: var(--text-main);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s ease;
            width: 100%;
            display: inline-flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            text-decoration: none;
        }

        .btn:hover {
            background: #1e293b;
            transform: translateY(-1px);
        }

        .btn:disabled {
            background: #cbd5e1;
            cursor: not-allowed;
            transform: none;
        }

        .btn-primary { background: var(--accent); }
        .btn-primary:hover { background: var(--accent-hover); }

        .btn-outline {
            background: transparent;
            color: var(--text-main);
            border: 1px solid var(--border);
        }
        .btn-outline:hover {
            background: #f1f5f9;
        }

        .flex-buttons {
            display: flex;
            gap: 12px;
            margin-top: 24px;
        }

        #file-info {
            display: none;
            margin-top: 16px;
            text-align: center;
            font-size: 0.875rem;
            color: var(--accent);
            font-weight: 500;
        }

        #process-section {
            display: none;
            margin-top: 24px;
        }

        /* Results view */
        #results-section {
            display: none;
            animation: fadeIn 0.4s ease forwards;
        }

        .comparison-layout {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 32px;
        }

        .image-card {
            display: flex;
            flex-direction: column;
            gap: 12px;
            background: #f8fafc;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid var(--border);
        }

        .image-card span {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .image-card img {
            width: 100%;
            border-radius: 8px;
            object-fit: contain;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 16px;
            margin-top: 32px;
            padding-top: 32px;
            border-top: 1px solid var(--border);
        }

        .metric {
            text-align: center;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 4px;
        }

        .metric-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Loader */
        .loader-container {
            display: none;
            flex-direction: column;
            align-items: center;
            padding: 40px 0;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(59, 130, 246, 0.1);
            border-radius: 50%;
            border-top-color: var(--accent);
            animation: spin 0.8s linear infinite;
            margin-bottom: 16px;
        }

        @keyframes spin { 100% { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

        @media (max-width: 768px) {
            .comparison-layout { grid-template-columns: 1fr; }
            .metrics-grid { grid-template-columns: repeat(3, 1fr); gap: 24px; }
            .flex-buttons { flex-direction: column; }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Atmospheric Dehazing</h1>
        <p>Advanced recovery modeling via Dark Channel Prior & Guided Filtering</p>
    </div>

    <!-- Upload Interface -->
    <div class="card" id="upload-card">
        <input type="file" id="imageInput" accept="image/*" style="display: none;">
        
        <div class="upload-area" id="drop-zone" onclick="document.getElementById('imageInput').click()">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            <div class="upload-text">Click to browse or drag image here</div>
            <div class="upload-hint">Supports JPG, PNG (Max 10MB)</div>
        </div>

        <div id="file-info">Selected: <span id="filename"></span></div>

        <div id="process-section">
            <button id="processBtn" class="btn btn-primary" onclick="processImage()">
                <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                Apply Dehazing Filter
            </button>
        </div>
    </div>

    <!-- Loader -->
    <div class="card loader-container" id="loader">
        <div class="spinner"></div>
        <div style="font-weight: 500; color: var(--text-main);">Processing Image...</div>
        <div style="font-size: 0.875rem; color: var(--text-muted); margin-top: 4px;">Computing transmission map and radiance recovery</div>
    </div>

    <!-- Results Interface -->
    <div class="card" id="results-section">
        <div class="comparison-layout">
            <div class="image-card">
                <span>Original</span>
                <img id="origDisplay" src="" alt="Original Image">
            </div>
            <div class="image-card">
                <span>Recovered</span>
                <img id="dehazedDisplay" src="" alt="Dehazed Image">
            </div>
        </div>

        <div class="flex-buttons">
            <a id="downloadBtn" class="btn btn-primary" download="dehazed_result.png">
                <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                Save Recovered Image
            </a>
            <button class="btn btn-outline" onclick="resetUI()">Upload Another</button>
        </div>

        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value" id="m-psnr">--</div>
                <div class="metric-label">PSNR</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="m-ssim">--</div>
                <div class="metric-label">SSIM</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="m-mse">--</div>
                <div class="metric-label">MSE</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="m-haze">--</div>
                <div class="metric-label">Haze Red.</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="m-contrast">--</div>
                <div class="metric-label">Contrast</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="m-detail">--</div>
                <div class="metric-label">Detail Pres.</div>
            </div>
        </div>
    </div>

</div>

<script>
    let uploadedImage = null;

    // File input handling
    document.getElementById('imageInput').addEventListener('change', function(e) {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    // Drag and drop handling
    const dropZone = document.getElementById('drop-zone');
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = 'var(--accent)'; });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.style.borderColor = ''; });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '';
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImage = e.target.result;
            document.getElementById('filename').textContent = file.name;
            document.getElementById('file-info').style.display = 'block';
            document.getElementById('process-section').style.display = 'block';
            
            // Set original image preview instantly
            document.getElementById('origDisplay').src = uploadedImage;
        };
        reader.readAsDataURL(file);
    }

    async function processImage() {
        if (!uploadedImage) return;

        // UI State: Loading
        document.getElementById('upload-card').style.display = 'none';
        document.getElementById('results-section').style.display = 'none';
        document.getElementById('loader').style.display = 'flex';

        try {
            const response = await fetch('/dehaze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: uploadedImage })
            });

            const result = await response.json();

            if (result.success) {
                // Populate results
                document.getElementById('dehazedDisplay').src = result.dehazed_image;
                document.getElementById('downloadBtn').href = result.dehazed_image;
                
                // Populate metrics
                document.getElementById('m-psnr').textContent = result.metrics.psnr;
                document.getElementById('m-ssim').textContent = result.metrics.ssim;
                document.getElementById('m-mse').textContent = result.metrics.mse;
                document.getElementById('m-haze').textContent = result.metrics.haze_reduction;
                document.getElementById('m-contrast').textContent = result.metrics.contrast_improvement;
                document.getElementById('m-detail').textContent = result.metrics.detail_preservation;

                // UI State: Show Results
                document.getElementById('loader').style.display = 'none';
                document.getElementById('results-section').style.display = 'block';
            } else {
                throw new Error(result.error || 'Server error occurred');
            }
        } catch (error) {
            alert('Failed to process image: ' + error.message);
            resetUI();
        }
    }

    function resetUI() {
        uploadedImage = null;
        document.getElementById('imageInput').value = '';
        document.getElementById('file-info').style.display = 'none';
        document.getElementById('process-section').style.display = 'none';
        document.getElementById('loader').style.display = 'none';
        document.getElementById('results-section').style.display = 'none';
        document.getElementById('upload-card').style.display = 'block';
    }
</script>

</body>
</html>'''
    
    with open('professional_dashboard.html', 'w') as f:
        f.write(html_content)

def start_professional_server():
    """Start the professional dehazing server"""
    PORT = 8084
    
    # Create the professional HTML dashboard
    create_professional_dashboard()
    
    # Start server
    with socketserver.TCPServer(("", PORT), ProfessionalDehazeRequestHandler) as httpd:
        print("PROFESSIONAL Atmospheric Dehazing Dashboard started!")
        print("Open your browser and go to: http://localhost:" + str(PORT))
        print("Using advanced atmospheric scattering model for quality haze removal")
        print("Server running on port " + str(PORT) + "...")
        print("Press Ctrl+C to stop the server")
        
        # Open browser automatically
        webbrowser.open('http://localhost:' + str(PORT))
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Professional server stopped by user")

if __name__ == "__main__":
    start_professional_server()
