# Enhanced Quantum DNA Analysis

A comprehensive quantum computing framework for DNA sequence similarity analysis using NEQR and FRQI encoding methods.

## 🚀 Features

- **Quantum DNA Encoding**: Implementation of NEQR and FRQI quantum encoding methods
- **Comprehensive Analysis**: Diversity, statistical, and scaling analysis modules
- **High Performance**: Optimized parallel processing with circuit caching
- **Statistical Rigor**: FDR correction, ANOVA analysis, and bootstrap confidence intervals
- **Professional Visualization**: Minimalistic IBM-style plots and 3D surfaces
- **Reproducible Research**: Comprehensive logging and seed management
- **Academic Quality**: Publication-ready results and documentation

## 📁 Project Structure

```
quantum-dna-analysis/
├── src/
│   ├── config/          # Configuration and settings
│   ├── core/            # Core quantum computing components
│   ├── analysis/        # Analysis modules (diversity, statistical, scaling)
│   ├── utils/           # Utility functions (sequence, stats, logging)
│   └── visualization/   # Plotting and visualization
├── main.py              # Main execution script
├── requirements.txt     # Dependencies
├── README.md           # This file
└── results/            # Output directory
    ├── figures/        # Generated plots
    ├── tables/         # CSV results
    ├── supplementary/  # Checkpoints and additional data
    └── logs/           # Execution logs
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/quantum-dna-analysis.git
   cd quantum-dna-analysis
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🔬 Usage

### Quick Start

Run the complete analysis suite:

```bash
python main.py
```

### Module Usage

#### 1. Diversity Analysis
```python
from src.core.runner import EnhancedCPURunner
from src.analysis.diversity import EnhancedDiversityAnalysis

runner = EnhancedCPURunner()
diversity_analyzer = EnhancedDiversityAnalysis(runner)
results = diversity_analyzer.analyze()
diversity_analyzer.plot_minimalistic_results()
```

#### 2. Statistical Analysis
```python
from src.analysis.statistical import EnhancedStatisticalAnalyzer

stat_analyzer = EnhancedStatisticalAnalyzer(runner)
results = stat_analyzer.analyze()
stat_analyzer.plot_correlation_analysis()
stat_analyzer.save_comprehensive_results()
```

#### 3. Scaling Analysis
```python
from src.analysis.scaling import EnhancedScalingAnalysis

scaling_analyzer = EnhancedScalingAnalysis(runner)
results = scaling_analyzer.analyze()
scaling_analyzer.plot_scaling_results()
```

## ⚙️ Configuration

Modify `src/config/settings.py` to customize:

- **Sequence parameters**: Length, GC content levels
- **Quantum settings**: Shots, trials, noise models
- **Analysis parameters**: Number of sequences, scaling ranges
- **Visualization**: Colors, DPI, styling

```python
CONFIG = {
    'shots': 8192,
    'sequence_length': 14,
    'gc_levels': [0.1, 0.3, 0.5, 0.7, 0.9],
    'figure_dpi': 600,
    # ... more options
}
```

## 📊 Analysis Types

### 1. Diversity Analysis
- Comprehensive test cases across similarity levels
- GC content analysis
- Statistical significance testing with FDR correction
- Bootstrap confidence intervals

### 2. Statistical Analysis
- ANOVA across GC content levels
- Correlation analysis
- Comprehensive statistical summaries
- 3D performance surfaces

### 3. Scaling Analysis
- Circuit complexity scaling
- Performance vs sequence length
- Resource utilization analysis
- Exponential growth modeling

## 🎨 Visualization

All plots use minimalistic IBM design principles:
- Clean, professional appearance
- Distinct color schemes for different GC levels
- Error bars and confidence intervals
- High-resolution PDF output

## 📈 Performance

- **Parallel Processing**: Utilizes all CPU cores
- **Circuit Caching**: Reduces redundant quantum circuit creation
- **Memory Management**: Automatic cleanup and optimization
- **Progress Tracking**: Real-time progress bars

## 📋 Output Files

### Results Tables (CSV)
- `diversity_results_enhanced.csv` - Comprehensive diversity analysis
- `statistical_results_enhanced.csv` - Statistical analysis results
- `scaling_results_enhanced.csv` - Scaling analysis results
- `performance_statistics.csv` - Execution performance metrics

### Figures (PDF)
- `diversity_comparison_enhanced.pdf` - Diversity analysis plots
- `correlation_analysis_enhanced.pdf` - Correlation analysis
- `gc_content_heatmap.pdf` - GC content heatmap
- `scaling_analysis_enhanced.pdf` - Scaling analysis plots

### Logs
- `comprehensive_execution_log.json` - Complete execution log
- Environment information and reproducibility data

## 🔧 Development

### Running Tests
```bash
pytest tests/ -v --cov=src/
```

### Code Formatting
```bash
black src/ main.py
flake8 src/ main.py
```

### Documentation
```bash
cd docs/
make html
```

## 🧬 Quantum DNA Encoding

### NEQR (Novel Enhanced Quantum Representation)
- Quantum superposition for position encoding
- Binary representation of nucleotides
- Swap test for similarity measurement

### FRQI (Flexible Representation of Quantum Images)
- Angle-based nucleotide encoding
- Quantum rotation gates
- Amplitude measurement for similarity

## 📖 Research Applications

This framework is designed for:
- Quantum bioinformatics research
- DNA sequence similarity analysis
- Quantum algorithm benchmarking
- Bioinformatics method comparison
- Academic publications and presentations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{quantum_dna_analysis,
  title={Enhanced Quantum DNA Analysis Framework},
  author={Quantum DNA Analysis Team},
  year={2024},
  url={https://github.com/yourusername/quantum-dna-analysis}
}
```

## 🆘 Support

For questions and support:
- Open an issue on GitHub
- Email: contact@quantumdna.com
- Documentation: [Link to docs]

## 🎯 Future Enhancements

- [ ] GPU acceleration support
- [ ] Real quantum hardware integration
- [ ] Additional encoding methods
- [ ] Interactive web interface
- [ ] Cloud deployment options
- [ ] Extended statistical tests
- [ ] Performance optimization

---

**Made with ❤️ for quantum bioinformatics research**