"""
Generate CrowdPulse Application Icon (.ico)
Creates a multi-resolution Windows icon for the desktop shortcut.
"""
from PIL import Image, ImageDraw
import os

def create_crowdpulse_icon(output_path="crowdpulse.ico"):
    base_size = 256
    img = Image.new("RGBA", (base_size, base_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. Background Rounded Disc (Deep Navy / Cyberpunk Night)
    bg_color = (15, 23, 42, 255)       # #0f172a
    border_color = (14, 165, 233, 255)  # #0ea5e9
    draw.ellipse([8, 8, base_size - 8, base_size - 8], fill=bg_color, outline=border_color, width=6)

    # 2. Concentric Radar / Sonar Rings (Semi-transparent cyan)
    draw.ellipse([36, 36, base_size - 36, base_size - 36], outline=(6, 182, 212, 80), width=3)
    draw.ellipse([64, 64, base_size - 64, base_size - 64], outline=(6, 182, 212, 120), width=3)
    draw.ellipse([92, 92, base_size - 92, base_size - 92], outline=(6, 182, 212, 160), width=2)

    # 3. Crowd Nodes (Center Cluster representing people/crowd density)
    # Center person (primary)
    draw.ellipse([116, 75, 140, 99], fill=(240, 253, 250, 255))  # Head
    draw.chord([102, 103, 154, 155], start=180, end=0, fill=(45, 212, 191, 255))  # Torso

    # Left person
    draw.ellipse([76, 92, 96, 112], fill=(204, 251, 241, 230))  # Head
    draw.chord([66, 116, 106, 156], start=180, end=0, fill=(20, 184, 166, 210))  # Torso

    # Right person
    draw.ellipse([160, 92, 180, 112], fill=(204, 251, 241, 230))  # Head
    draw.chord([150, 116, 190, 156], start=180, end=0, fill=(20, 184, 166, 210))  # Torso

    # 4. Dynamic Vital Pulse Wave (ECG / Radar Pulse across the lower half)
    pulse_points = [
        (20, 172),
        (75, 172),
        (95, 172),
        (108, 142),
        (120, 206),
        (134, 128),
        (148, 194),
        (160, 172),
        (180, 172),
        (236, 172),
    ]
    # Draw glow underneath
    draw.line(pulse_points, fill=(6, 182, 212, 100), width=9, joint="curve")
    # Draw sharp pulse line
    draw.line(pulse_points, fill=(56, 189, 248, 255), width=5, joint="curve")
    draw.line(pulse_points, fill=(255, 255, 255, 240), width=2, joint="curve")

    # 5. Glowing alert indicator dot at pulse apex
    draw.ellipse([130, 124, 138, 132], fill=(239, 68, 68, 255), outline=(255, 255, 255, 255), width=2)

    # Save as multi-resolution Windows Icon (.ico)
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    icon_images = [img.resize(s, Image.Resampling.LANCZOS) for s in sizes]
    
    icon_images[0].save(
        output_path,
        format="ICO",
        sizes=sizes,
        append_images=icon_images[1:]
    )
    print(f"[OK] CrowdPulse multi-resolution icon generated at: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    create_crowdpulse_icon("crowdpulse.ico")
