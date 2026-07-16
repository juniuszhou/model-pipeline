"""Video model entry (legacy filename).

- Image ViT:   ``mymodel.vit``
- Video ViT:   ``mymodel.video_vit``  (efficient space-time attention)

Prefer::

    python -m mymodel.video_vit
"""

try:
    from video_vit import VideoViT, main, smoke_test
except ImportError:
    from mymodel.video_vit import VideoViT, main, smoke_test

__all__ = ["VideoViT", "smoke_test", "main"]

if __name__ == "__main__":
    main()
