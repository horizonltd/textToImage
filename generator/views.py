import os
import uuid
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model once at startup
model_path = "CompVis/stable-diffusion-v1-4"  # Change if using a different model
pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

@csrf_exempt  # Disable CSRF for testing (Enable proper CSRF handling in production)
def generate_image(request):
    if request.method == "GET":
        return render(request, "generate.html")  # ✅ Show the form instead of an error

    if request.method == "POST":
        text_prompt = request.POST.get("prompt", None)  # Get text input

        if not text_prompt:
            return JsonResponse({"error": "No prompt provided"}, status=400)

        try:
            # Generate unique filename
            image_name = f"{uuid.uuid4().hex}.png"
            image_path = os.path.join(settings.MEDIA_ROOT, "generated_images", image_name)

            # Generate image from text
            image = pipeline(text_prompt).images[0]
            image.save(image_path)  # Save to MEDIA folder

            # Return image URL
            image_url = f"{settings.MEDIA_URL}generated_images/{image_name}"
            return JsonResponse({"image_url": image_url})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)
