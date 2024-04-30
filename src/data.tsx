// Project title
export const title = "Invisible Stitch: Generating Smooth 3D Scenes with Depth Inpainting"

export const description = "We introduce a depth completion network that allows generating geometrically coherent 3D scenes, and present a new benchmarking scheme for scene generation methods that is based on ground truth geometry."

// Abstract
export const abstract = "3D scene generation has quickly become a challenging new research direction, fueled by consistent improvements of 2D generative diffusion models. Most prior work in this area generates scenes by iteratively stitching newly generated frames with existing geometry.  These works often depend on pre-trained monocular depth estimators to lift the generated images into 3D, fusing them with the existing scene representation. These approaches are then often evaluated via a text metric, measuring the similarity between the generated images and a given text prompt. In this work, we make two fundamental contributions to the field of 3D scene generation. First, we note that lifting images to 3D with a monocular depth estimation model is suboptimal as it ignores the geometry of the existing scene. We thus introduce a novel depth completion model, trained via teacher distillation and self-training to learn the 3D fusion process, resulting in improved geometric coherence of the scene. Second, we introduce a new benchmarking scheme for scene generation methods that is based on ground truth geometry, and thus measures the quality of the structure of the scene."

// Authors
export const authors = [
  {
    'name': 'Paul Engstler',
    'institutions': ["University of Oxford"],
    'url': "https://paulengstler.com",
    'tags': []
  },
  {
    'name': 'Andrea Vedaldi',
    'institutions': ["University of Oxford"],
    'url': "https://www.robots.ox.ac.uk/~vedaldi/",
    'tags': []
  },
  {
    'name': 'Iro Laina',
    'institutions': ["University of Oxford"],
    'url': "http://campar.in.tum.de/Main/IroLaina",
    'tags': []
  },
  {
    'name': 'Christian Rupprecht',
    'institutions': ["University of Oxford"],
    'url': "https://chrirupp.github.io/",
    'tags': []
  },
]

export const author_tags = [

]

// Links
export const links = {
  'paper': "#",
  'github': "https://github.com/paulengstler/invisible-stitch",
  'demo': 'https://huggingface.co/spaces/paulengstler/invisible-stitch',
}

// Acknowledgements
export const acknowledgements = "P. E., A. V., I. L., and C.R. are supported by ERC-UNION- CoG-101001212. P.E. is also supported by Meta Research. I.L. and C.R. also receive support from VisualAI EP/T028572/1."

// Citations
export const citationId = "engstler2024invisible"
export const citationAuthors = "Paul Engstler and Andrea Vedaldi and Iro Laina and Christian Rupprecht"
export const citationYear = "2024"
export const citationBooktitle = "Arxiv"