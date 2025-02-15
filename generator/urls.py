from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import generate_image

urlpatterns = [
    path("generate/", generate_image, name="generate_image"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
