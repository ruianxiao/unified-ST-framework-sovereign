# Slide Generation Scripts

This directory contains Python scripts for generating presentation slides from research analysis results.

## Current Scripts

### Transformation Analysis Slide
`create_transformation_slide.py` generates a PowerPoint slide summarizing the macro variable transformation analysis.

## Setup

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the script:
```bash
python create_transformation_slide.py
```

The script will create a PowerPoint file at `Slides/output/transformation_analysis.pptx`

## Output

The generated slide includes:
- Title and subtitle
- Complete table of macro variables and their transformation results
- Key findings
- Economic theory alignment
- Technical notes

## Customization

To modify the slide:
1. Edit the data dictionary in `create_transformation_slide.py`
2. Adjust formatting parameters (font sizes, positions, etc.)
3. Run the script again to generate an updated slide

## Dependencies

- python-pptx: For PowerPoint file generation
- pandas: For data handling 