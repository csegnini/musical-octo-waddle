"""
Statistical Analysis Module Information Module.

This module provides comprehensive information about the statistical analysis module
capabilities, features, and usage guidelines for classical and advanced statistical methods.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive statistical analysis module information.
    
    Returns:
        Dictionary containing complete module details
    """
    return {
        'module_name': 'Advanced Statistical Analysis Framework',
        'version': '1.0.0',
        'description': 'Comprehensive statistical analysis framework with classical and modern statistical methods for hypothesis testing, regression analysis, and statistical modeling',
        'author': 'AI ML Platform Team',
        'license': 'MIT',
        'created_date': '2025-07-27',
        'last_updated': datetime.now().isoformat(),
        'core_components': {
            'statistical_models': {
                'file': '__init__.py',
                'lines_of_code': 900,
                'description': 'Advanced statistical models with comprehensive testing and analysis capabilities',
                'key_classes': ['StatisticalModel', 'StatisticalConfig', 'TestType', 'DistributionType'],
                'features': [
                    '12+ statistical methods (T-test, ANOVA, Chi-square, Regression, Bayesian, Non-parametric)',
                    'Automatic assumption testing and validation',
                    'Robust statistical inference with confidence intervals',
                    'Effect size calculations and power analysis',
                    'Multiple comparison corrections (Bonferroni, FDR, Holm)',
                    'Bootstrap and permutation testing',
                    'Advanced regression diagnostics',
                    'Comprehensive statistical reporting'
                ]
            },
            'base_integration': {
                'file': '../base/__init__.py',
                'description': 'BaseModel interface compliance for consistent API',
                'key_classes': ['BaseModel', 'ModelMetadata', 'ModelType', 'ProblemType', 'ModelStatus'],
                'features': [
                    'Standardized fit/predict interface',
                    'Model lifecycle management',
                    'Metadata tracking and versioning',
                    'Status monitoring and error handling',
                    'Type-safe implementations'
                ]
            }
        },
        'supported_methods': {
            'hypothesis_testing': {
                't_tests': {
                    'description': 'Student\'s t-tests for mean comparisons',
                    'variants': ['One-sample', 'Two-sample independent', 'Paired samples'],
                    'assumptions': ['Normality', 'Independence', 'Equal variances (for two-sample)'],
                    'use_cases': ['Mean comparison', 'Treatment effect testing', 'Before/after studies']
                },
                'anova': {
                    'description': 'Analysis of Variance for multiple group comparisons',
                    'variants': ['One-way ANOVA', 'Two-way ANOVA', 'Repeated measures ANOVA', 'ANCOVA'],
                    'assumptions': ['Normality', 'Independence', 'Homogeneity of variance'],
                    'use_cases': ['Multiple group comparison', 'Factorial designs', 'Treatment interactions']
                },
                'chi_square': {
                    'description': 'Chi-square tests for categorical data analysis',
                    'variants': ['Goodness of fit', 'Independence test', 'Homogeneity test'],
                    'assumptions': ['Expected frequencies ≥ 5', 'Independence of observations'],
                    'use_cases': ['Categorical associations', 'Distribution testing', 'Contingency analysis']
                },
                'non_parametric': {
                    'description': 'Distribution-free statistical tests',
                    'variants': ['Mann-Whitney U', 'Wilcoxon signed-rank', 'Kruskal-Wallis', 'Friedman test'],
                    'assumptions': ['Independence', 'Ordinal or continuous data'],
                    'use_cases': ['Non-normal data', 'Small samples', 'Robust analysis']
                }
            },
            'regression_analysis': {
                'linear_regression': {
                    'description': 'Linear relationship modeling between variables',
                    'variants': ['Simple linear', 'Multiple regression', 'Polynomial regression'],
                    'assumptions': ['Linearity', 'Independence', 'Homoscedasticity', 'Normality of residuals'],
                    'use_cases': ['Prediction modeling', 'Relationship quantification', 'Causal inference']
                },
                'logistic_regression': {
                    'description': 'Regression for binary and categorical outcomes',
                    'variants': ['Binary logistic', 'Multinomial logistic', 'Ordinal logistic'],
                    'assumptions': ['Independence', 'Linear relationship with log-odds', 'No multicollinearity'],
                    'use_cases': ['Binary classification', 'Probability modeling', 'Risk factor analysis']
                },
                'robust_regression': {
                    'description': 'Regression methods resistant to outliers',
                    'variants': ['Huber regression', 'RANSAC', 'Theil-Sen estimator'],
                    'assumptions': ['Minimal assumptions about error distribution'],
                    'use_cases': ['Outlier-resistant modeling', 'Robust prediction', 'Contaminated data']
                },
                'time_series_regression': {
                    'description': 'Regression for temporal data with autocorrelation',
                    'variants': ['AR models', 'MA models', 'ARMA models', 'VAR models'],
                    'assumptions': ['Stationarity', 'No unit roots', 'Appropriate lag structure'],
                    'use_cases': ['Time series prediction', 'Economic modeling', 'Trend analysis']
                }
            },
            'bayesian_methods': {
                'bayesian_inference': {
                    'description': 'Bayesian approach to statistical inference',
                    'variants': ['Bayesian t-test', 'Bayesian ANOVA', 'Bayesian regression'],
                    'assumptions': ['Prior distribution specification', 'Likelihood function'],
                    'use_cases': ['Uncertainty quantification', 'Prior knowledge incorporation', 'Small sample analysis']
                },
                'mcmc_sampling': {
                    'description': 'Markov Chain Monte Carlo methods for posterior sampling',
                    'variants': ['Metropolis-Hastings', 'Gibbs sampling', 'NUTS sampling'],
                    'assumptions': ['Convergence of chains', 'Adequate burn-in period'],
                    'use_cases': ['Complex posterior distributions', 'Hierarchical models', 'Bayesian model averaging']
                }
            },
            'multivariate_analysis': {
                'manova': {
                    'description': 'Multivariate Analysis of Variance',
                    'variants': ['One-way MANOVA', 'Two-way MANOVA', 'Repeated measures MANOVA'],
                    'assumptions': ['Multivariate normality', 'Homogeneity of covariance matrices', 'Independence'],
                    'use_cases': ['Multiple dependent variables', 'Multivariate group comparisons', 'Profile analysis']
                },
                'factor_analysis': {
                    'description': 'Exploratory and confirmatory factor analysis',
                    'variants': ['EFA', 'CFA', 'Principal axis factoring'],
                    'assumptions': ['Linear relationships', 'Adequate sample size', 'Factorability'],
                    'use_cases': ['Dimension reduction', 'Latent variable identification', 'Scale validation']
                },
                'canonical_correlation': {
                    'description': 'Analysis of relationships between variable sets',
                    'variants': ['Classical canonical correlation', 'Regularized CCA'],
                    'assumptions': ['Multivariate normality', 'Linear relationships'],
                    'use_cases': ['Variable set relationships', 'Multivariate association', 'Dimension reduction']
                }
            }
        },
        'statistical_tests_catalog': {
            'parametric_tests': {
                'description': 'Tests assuming specific probability distributions',
                'tests': {
                    'one_sample_t_test': {
                        'purpose': 'Test if sample mean differs from population mean',
                        'null_hypothesis': 'Sample mean equals population mean',
                        'test_statistic': 't = (x̄ - μ) / (s / √n)',
                        'degrees_of_freedom': 'n - 1',
                        'effect_size': 'Cohen\'s d'
                    },
                    'independent_t_test': {
                        'purpose': 'Compare means of two independent groups',
                        'null_hypothesis': 'Group means are equal',
                        'test_statistic': 't = (x̄₁ - x̄₂) / sp√(1/n₁ + 1/n₂)',
                        'degrees_of_freedom': 'n₁ + n₂ - 2',
                        'effect_size': 'Cohen\'s d, Glass\'s Δ'
                    },
                    'paired_t_test': {
                        'purpose': 'Compare paired observations',
                        'null_hypothesis': 'Mean difference equals zero',
                        'test_statistic': 't = d̄ / (sd / √n)',
                        'degrees_of_freedom': 'n - 1',
                        'effect_size': 'Cohen\'s d for paired samples'
                    },
                    'one_way_anova': {
                        'purpose': 'Compare means across multiple groups',
                        'null_hypothesis': 'All group means are equal',
                        'test_statistic': 'F = MSB / MSW',
                        'degrees_of_freedom': 'k-1, N-k',
                        'effect_size': 'η² (eta-squared), ω² (omega-squared)'
                    }
                }
            },
            'non_parametric_tests': {
                'description': 'Distribution-free tests with minimal assumptions',
                'tests': {
                    'mann_whitney_u': {
                        'purpose': 'Compare two independent groups (alternative to t-test)',
                        'null_hypothesis': 'Distributions are identical',
                        'test_statistic': 'U statistic based on rank sums',
                        'effect_size': 'r = Z / √N, Cliff\'s delta'
                    },
                    'wilcoxon_signed_rank': {
                        'purpose': 'Compare paired observations (alternative to paired t-test)',
                        'null_hypothesis': 'Median difference equals zero',
                        'test_statistic': 'W = sum of positive signed ranks',
                        'effect_size': 'r = Z / √N'
                    },
                    'kruskal_wallis': {
                        'purpose': 'Compare multiple groups (alternative to ANOVA)',
                        'null_hypothesis': 'All groups have same distribution',
                        'test_statistic': 'H = 12/N(N+1) × Σ(Ri²/ni) - 3(N+1)',
                        'effect_size': 'η² based on ranks'
                    },
                    'friedman_test': {
                        'purpose': 'Compare repeated measures across conditions',
                        'null_hypothesis': 'No systematic differences across conditions',
                        'test_statistic': 'χ² = 12/nk(k+1) × Σ(Ri²) - 3n(k+1)',
                        'effect_size': 'Kendall\'s W'
                    }
                }
            },
            'categorical_tests': {
                'description': 'Tests for categorical and count data',
                'tests': {
                    'chi_square_goodness': {
                        'purpose': 'Test if sample follows expected distribution',
                        'null_hypothesis': 'Sample follows expected distribution',
                        'test_statistic': 'χ² = Σ[(Observed - Expected)² / Expected]',
                        'degrees_of_freedom': 'categories - 1',
                        'effect_size': 'Cramér\'s V, φ (phi)'
                    },
                    'chi_square_independence': {
                        'purpose': 'Test independence of two categorical variables',
                        'null_hypothesis': 'Variables are independent',
                        'test_statistic': 'χ² = Σ[(Observed - Expected)² / Expected]',
                        'degrees_of_freedom': '(rows - 1) × (columns - 1)',
                        'effect_size': 'Cramér\'s V, contingency coefficient'
                    },
                    'fishers_exact': {
                        'purpose': 'Test independence in 2×2 tables (small samples)',
                        'null_hypothesis': 'Variables are independent',
                        'test_statistic': 'Exact probability calculation',
                        'effect_size': 'Odds ratio, φ (phi) coefficient'
                    }
                }
            }
        },
        'advanced_features': {
            'power_analysis': {
                'description': 'Statistical power and sample size calculations',
                'capabilities': {
                    'prospective_analysis': 'Calculate required sample size for desired power',
                    'retrospective_analysis': 'Calculate achieved power for given sample size',
                    'sensitivity_analysis': 'Determine detectable effect size',
                    'optimal_design': 'Find optimal allocation of resources'
                },
                'supported_tests': ['t-tests', 'ANOVA', 'Chi-square', 'Correlation', 'Regression'],
                'power_curves': 'Visualization of power vs. effect size relationships'
            },
            'effect_size_measures': {
                'description': 'Standardized measures of practical significance',
                'categories': {
                    'standardized_differences': ['Cohen\'s d', 'Glass\'s Δ', 'Hedges\' g'],
                    'variance_explained': ['η² (eta-squared)', 'ω² (omega-squared)', 'R²'],
                    'correlation_based': ['r (correlation)', 'R² (coefficient of determination)'],
                    'non_parametric': ['Cliff\'s delta', 'Vargha-Delaney A', 'rank-biserial correlation']
                },
                'interpretation_guidelines': 'Cohen\'s conventions and field-specific benchmarks'
            },
            'assumption_testing': {
                'description': 'Automated testing of statistical assumptions',
                'normality_tests': ['Shapiro-Wilk', 'Anderson-Darling', 'Kolmogorov-Smirnov', 'Jarque-Bera'],
                'homogeneity_tests': ['Levene\'s test', 'Bartlett\'s test', 'Brown-Forsythe test'],
                'independence_tests': ['Durbin-Watson', 'Ljung-Box', 'Autocorrelation function'],
                'outlier_detection': ['Z-score method', 'IQR method', 'Modified Z-score', 'Isolation Forest']
            },
            'multiple_comparisons': {
                'description': 'Correction methods for multiple testing',
                'methods': {
                    'family_wise_error_rate': ['Bonferroni', 'Holm-Bonferroni', 'Šidák correction'],
                    'false_discovery_rate': ['Benjamini-Hochberg', 'Benjamini-Yekutieli'],
                    'resampling_based': ['Permutation-based corrections', 'Bootstrap-based corrections']
                },
                'post_hoc_tests': ['Tukey HSD', 'Scheffe\'s test', 'Dunnett\'s test', 'Games-Howell']
            },
            'robust_methods': {
                'description': 'Statistical methods resistant to assumption violations',
                'robust_estimators': ['Huber M-estimators', 'Tukey\'s biweight', 'Median-based estimators'],
                'robust_tests': ['Welch\'s t-test', 'Brown-Forsythe ANOVA', 'Robust regression'],
                'resampling_methods': ['Bootstrap confidence intervals', 'Permutation tests', 'Jackknife procedures']
            },
            'bayesian_framework': {
                'description': 'Bayesian approach to statistical analysis',
                'prior_specifications': ['Informative priors', 'Non-informative priors', 'Conjugate priors'],
                'posterior_analysis': ['Credible intervals', 'Posterior predictive checks', 'Model comparison'],
                'mcmc_diagnostics': ['Trace plots', 'R-hat statistics', 'Effective sample size', 'Autocorrelation']
            }
        },
        'supported_problem_types': [
            {
                'name': 'Hypothesis Testing',
                'description': 'Testing statistical hypotheses about population parameters',
                'methods': ['t-tests', 'ANOVA', 'Chi-square', 'Non-parametric tests'],
                'outputs': ['Test statistic', 'p-value', 'Effect size', 'Confidence intervals'],
                'typical_applications': 'A/B testing, treatment comparisons, quality control'
            },
            {
                'name': 'Relationship Analysis',
                'description': 'Examining associations between variables',
                'methods': ['Correlation analysis', 'Regression analysis', 'Canonical correlation'],
                'outputs': ['Correlation coefficients', 'Regression coefficients', 'R-squared values'],
                'typical_applications': 'Predictive modeling, causal inference, variable selection'
            },
            {
                'name': 'Group Comparison',
                'description': 'Comparing means, medians, or distributions across groups',
                'methods': ['ANOVA', 'MANOVA', 'Non-parametric tests', 'Post-hoc analysis'],
                'outputs': ['F-statistics', 'Effect sizes', 'Multiple comparison results'],
                'typical_applications': 'Experimental design analysis, market research, clinical trials'
            },
            {
                'name': 'Distribution Analysis',
                'description': 'Analyzing and comparing probability distributions',
                'methods': ['Goodness-of-fit tests', 'Distribution fitting', 'Q-Q plots'],
                'outputs': ['Distribution parameters', 'Goodness-of-fit statistics', 'AIC/BIC values'],
                'typical_applications': 'Quality control, risk assessment, modeling assumptions'
            },
            {
                'name': 'Time Series Analysis',
                'description': 'Statistical analysis of temporal data',
                'methods': ['Trend analysis', 'Seasonality detection', 'Autocorrelation analysis'],
                'outputs': ['Trend coefficients', 'Seasonal components', 'Forecasts'],
                'typical_applications': 'Economic forecasting, monitoring systems, longitudinal studies'
            }
        ],
        'performance_metrics': {
            'test_validity': [
                'Type I Error Rate (α)', 'Type II Error Rate (β)', 'Statistical Power (1-β)',
                'False Discovery Rate', 'Family-wise Error Rate'
            ],
            'effect_size_metrics': [
                'Cohen\'s d', 'Eta-squared (η²)', 'Omega-squared (ω²)', 'R-squared (R²)',
                'Cramér\'s V', 'Phi coefficient (φ)', 'Cliff\'s delta'
            ],
            'model_fit_metrics': [
                'Akaike Information Criterion (AIC)', 'Bayesian Information Criterion (BIC)',
                'Log-likelihood', 'Deviance', 'Pseudo R-squared'
            ],
            'diagnostic_metrics': [
                'Residual Analysis', 'Cook\'s Distance', 'Leverage Values', 'DFBeta',
                'Durbin-Watson Statistic', 'Variance Inflation Factor (VIF)'
            ],
            'robustness_metrics': [
                'Breakdown Point', 'Influence Function', 'Efficiency Relative to OLS',
                'Bootstrap Confidence Interval Coverage'
            ]
        },
        'technical_specifications': {
            'performance': {
                'computation_speed': 'Optimized algorithms for large datasets (up to 1M+ observations)',
                'memory_efficiency': 'Efficient memory management for complex statistical procedures',
                'numerical_stability': 'Robust numerical algorithms with precision control',
                'scalability': 'Parallel processing for computationally intensive methods'
            },
            'accuracy_precision': {
                'numerical_precision': 'Double-precision floating-point arithmetic',
                'convergence_criteria': 'Adaptive convergence thresholds for iterative algorithms',
                'monte_carlo_accuracy': 'Controlled random number generation with reproducible seeds',
                'approximation_quality': 'High-quality approximations for complex distributions'
            },
            'software_compatibility': {
                'python_version': '3.7+',
                'core_dependencies': ['scipy', 'statsmodels', 'pandas', 'numpy'],
                'optional_dependencies': ['pymc3', 'arviz', 'seaborn', 'matplotlib'],
                'r_integration': 'Optional R interface through rpy2',
                'os_support': ['Windows', 'Linux', 'macOS']
            },
            'data_compatibility': {
                'input_formats': ['pandas DataFrames', 'numpy arrays', 'CSV files', 'Excel files'],
                'data_types': ['Numerical', 'Categorical', 'Ordinal', 'Binary', 'Time series'],
                'missing_data': 'Automatic handling with multiple imputation options',
                'data_validation': 'Comprehensive data quality checks and warnings'
            }
        },
        'integration_capabilities': {
            'statistical_software': {
                'scipy_stats': 'Native SciPy integration for basic statistical functions',
                'statsmodels': 'Advanced statistical modeling capabilities',
                'scikit_learn': 'Machine learning pipeline compatibility',
                'r_interface': 'Optional R integration for specialized statistical procedures'
            },
            'visualization': {
                'statistical_plots': 'Comprehensive statistical visualization library',
                'diagnostic_plots': 'Automated residual analysis and assumption checking plots',
                'effect_plots': 'Effect size visualization and confidence interval plots',
                'interactive_dashboards': 'Streamlit/Dash integration for interactive analysis'
            },
            'reporting': {
                'automated_reports': 'Statistical analysis reports in multiple formats',
                'apa_formatting': 'APA-style statistical reporting',
                'latex_output': 'Publication-ready LaTeX tables and equations',
                'web_reports': 'HTML reports with interactive elements'
            }
        },
        'validation_framework': {
            'assumption_validation': {
                'description': 'Comprehensive testing of statistical assumptions',
                'normality_assessment': 'Multiple normality tests with visual diagnostics',
                'homogeneity_assessment': 'Variance equality testing across groups',
                'independence_assessment': 'Autocorrelation and independence testing',
                'linearity_assessment': 'Relationship linearity evaluation'
            },
            'robustness_checks': {
                'description': 'Sensitivity analysis and robustness evaluation',
                'outlier_sensitivity': 'Analysis of outlier influence on results',
                'assumption_relaxation': 'Results under different assumption sets',
                'bootstrap_validation': 'Bootstrap-based validation of statistical results',
                'cross_validation': 'Statistical model validation using resampling'
            },
            'reproducibility': {
                'description': 'Ensuring reproducible statistical analysis',
                'random_seed_control': 'Controlled randomization for reproducible results',
                'version_tracking': 'Analysis version control and change tracking',
                'computational_environment': 'Environment specification for reproducibility',
                'audit_trail': 'Complete analysis audit trail and documentation'
            }
        }
    }


def get_method_comparison() -> Dict[str, Dict[str, Any]]:
    """Get detailed comparison of statistical methods."""
    return {
        'parametric_vs_nonparametric': {
            'parametric_methods': {
                'assumptions': ['Normality', 'Homogeneity of variance', 'Independence'],
                'advantages': ['More powerful when assumptions met', 'Precise parameter estimates', 'Well-established theory'],
                'disadvantages': ['Sensitive to assumption violations', 'May be inappropriate for non-normal data'],
                'examples': ['t-tests', 'ANOVA', 'Pearson correlation', 'Linear regression'],
                'when_to_use': 'Large samples, normal distributions, assumptions clearly met'
            },
            'nonparametric_methods': {
                'assumptions': ['Independence', 'Ordinal or continuous data'],
                'advantages': ['Robust to outliers', 'No distributional assumptions', 'Works with small samples'],
                'disadvantages': ['Less powerful when parametric assumptions met', 'Limited parameter interpretation'],
                'examples': ['Mann-Whitney U', 'Kruskal-Wallis', 'Spearman correlation', 'Robust regression'],
                'when_to_use': 'Non-normal data, small samples, presence of outliers, ordinal data'
            }
        },
        'frequentist_vs_bayesian': {
            'frequentist_approach': {
                'philosophy': 'Probability as long-run frequency of events',
                'advantages': ['Objective interpretation', 'Well-established procedures', 'Fast computation'],
                'disadvantages': ['No probability on parameters', 'Multiple testing issues', 'No prior knowledge incorporation'],
                'outputs': ['p-values', 'Confidence intervals', 'Test statistics'],
                'interpretation': 'Results interpreted in terms of repeated sampling'
            },
            'bayesian_approach': {
                'philosophy': 'Probability as degree of belief',
                'advantages': ['Probability on parameters', 'Prior knowledge incorporation', 'Natural uncertainty quantification'],
                'disadvantages': ['Subjective priors', 'Computational complexity', 'Learning curve'],
                'outputs': ['Posterior distributions', 'Credible intervals', 'Bayes factors'],
                'interpretation': 'Direct probability statements about parameters'
            }
        },
        'univariate_vs_multivariate': {
            'univariate_methods': {
                'scope': 'Single outcome variable analysis',
                'advantages': ['Simple interpretation', 'Fast computation', 'Clear results'],
                'disadvantages': ['Ignores variable relationships', 'Multiple testing when applied repeatedly'],
                'examples': ['One-sample t-test', 'Chi-square goodness of fit', 'Simple regression'],
                'typical_use': 'Single variable questions, exploratory analysis, simple comparisons'
            },
            'multivariate_methods': {
                'scope': 'Multiple outcome variables simultaneously',
                'advantages': ['Accounts for variable relationships', 'Controls family-wise error', 'More powerful'],
                'disadvantages': ['Complex interpretation', 'Large sample requirements', 'Computational complexity'],
                'examples': ['MANOVA', 'Multiple regression', 'Factor analysis', 'Canonical correlation'],
                'typical_use': 'Complex relationships, multiple outcomes, dimension reduction'
            }
        }
    }


def get_implementation_examples() -> Dict[str, str]:
    """Get comprehensive implementation examples for common statistical analyses."""
    return {
        'basic_hypothesis_testing': '''
# Basic Hypothesis Testing Example
from modeling.statistical_analysis import StatisticalModel, StatisticalConfig, TestType

# Configure t-test analysis
config = StatisticalConfig(
    test_type=TestType.INDEPENDENT_T_TEST,
    alpha_level=0.05,
    alternative='two-sided',
    equal_variances=True,
    effect_size_measure='cohens_d'
)

# Create and run the analysis
model = StatisticalModel(config, name="Treatment_Comparison")
results = model.fit(group1_data, group2_data)

# Get comprehensive results
print(f"Test statistic: {results.test_statistic:.3f}")
print(f"p-value: {results.p_value:.4f}")
print(f"Effect size (Cohen's d): {results.effect_size:.3f}")
print(f"95% CI: [{results.confidence_interval[0]:.3f}, {results.confidence_interval[1]:.3f}]")
print(f"Power: {results.power:.3f}")

# Assumption checking
assumption_results = model.check_assumptions()
print(f"Normality test p-value: {assumption_results['normality_p']:.4f}")
print(f"Equal variances test p-value: {assumption_results['levene_p']:.4f}")
''',

        'anova_analysis': '''
# One-way ANOVA with Post-hoc Analysis
from modeling.statistical_analysis import StatisticalModel, StatisticalConfig, TestType

# Configure ANOVA
config = StatisticalConfig(
    test_type=TestType.ONE_WAY_ANOVA,
    alpha_level=0.05,
    post_hoc_method='tukey_hsd',
    multiple_comparison_correction='bonferroni',
    effect_size_measure='eta_squared'
)

# Run ANOVA analysis
model = StatisticalModel(config, name="Multi_Group_Analysis")
results = model.fit(data, group_variable='treatment', dependent_variable='outcome')

# ANOVA results
print(f"F-statistic: {results.f_statistic:.3f}")
print(f"p-value: {results.p_value:.4f}")
print(f"Effect size (η²): {results.eta_squared:.3f}")

# Post-hoc comparisons
for comparison in results.post_hoc_results:
    print(f"{comparison.group1} vs {comparison.group2}: "
          f"p = {comparison.p_value:.4f}, "
          f"Cohen's d = {comparison.effect_size:.3f}")

# Assumption diagnostics
diagnostics = model.get_diagnostics()
print(f"Levene's test p-value: {diagnostics['levene_test']:.4f}")
print(f"Shapiro-Wilk test p-value: {diagnostics['shapiro_test']:.4f}")
''',

        'regression_analysis': '''
# Multiple Linear Regression with Diagnostics
from modeling.statistical_analysis import StatisticalModel, StatisticalConfig, TestType

# Configure regression analysis
config = StatisticalConfig(
    test_type=TestType.MULTIPLE_REGRESSION,
    alpha_level=0.05,
    confidence_level=0.95,
    standardize_predictors=True,
    include_interactions=False
)

# Run regression analysis
model = StatisticalModel(config, name="Predictive_Model")
results = model.fit(X_predictors, y_outcome)

# Model summary
print(f"R-squared: {results.r_squared:.3f}")
print(f"Adjusted R-squared: {results.adj_r_squared:.3f}")
print(f"F-statistic: {results.f_statistic:.3f}, p = {results.f_pvalue:.4f}")

# Coefficient results
for var, coef in results.coefficients.items():
    print(f"{var}: β = {coef.estimate:.3f}, "
          f"SE = {coef.std_error:.3f}, "
          f"t = {coef.t_value:.3f}, "
          f"p = {coef.p_value:.4f}")

# Regression diagnostics
diagnostics = model.get_diagnostics()
print(f"Durbin-Watson: {diagnostics['durbin_watson']:.3f}")
print(f"Breusch-Pagan test p-value: {diagnostics['breusch_pagan_p']:.4f}")

# Influential observations
influential = model.identify_outliers()
print(f"High leverage points: {len(influential['high_leverage'])}")
print(f"High Cook's distance: {len(influential['high_cooks'])}")
''',

        'nonparametric_analysis': '''
# Non-parametric Analysis Example
from modeling.statistical_analysis import StatisticalModel, StatisticalConfig, TestType

# Configure Mann-Whitney U test
config = StatisticalConfig(
    test_type=TestType.MANN_WHITNEY_U,
    alpha_level=0.05,
    alternative='two-sided',
    effect_size_measure='rank_biserial_correlation'
)

# Run non-parametric test
model = StatisticalModel(config, name="Robust_Comparison")
results = model.fit(group1_data, group2_data)

print(f"U-statistic: {results.test_statistic:.1f}")
print(f"p-value: {results.p_value:.4f}")
print(f"Effect size (r): {results.effect_size:.3f}")

# Bootstrap confidence interval for median difference
bootstrap_ci = model.bootstrap_confidence_interval(
    statistic='median_difference',
    n_bootstrap=10000,
    confidence_level=0.95
)
print(f"Bootstrap 95% CI for median difference: "
      f"[{bootstrap_ci[0]:.3f}, {bootstrap_ci[1]:.3f}]")

# Permutation test for additional validation
perm_p = model.permutation_test(n_permutations=10000)
print(f"Permutation test p-value: {perm_p:.4f}")
''',

        'bayesian_analysis': '''
# Bayesian Analysis Example
from modeling.statistical_analysis import StatisticalModel, StatisticalConfig, TestType

# Configure Bayesian t-test
config = StatisticalConfig(
    test_type=TestType.BAYESIAN_T_TEST,
    prior_type='default_cauchy',
    prior_scale=0.707,  # Default Cauchy scale
    credible_interval=0.95,
    n_samples=10000,
    n_chains=4
)

# Run Bayesian analysis
model = StatisticalModel(config, name="Bayesian_Comparison")
results = model.fit(group1_data, group2_data)

# Bayesian results
print(f"Bayes Factor (BF₁₀): {results.bayes_factor:.3f}")
print(f"Posterior median effect size: {results.posterior_median:.3f}")
print(f"95% Credible interval: "
      f"[{results.credible_interval[0]:.3f}, {results.credible_interval[1]:.3f}]")
print(f"Probability of positive effect: {results.prob_positive:.3f}")

# MCMC diagnostics
diagnostics = model.get_mcmc_diagnostics()
print(f"R-hat: {diagnostics['r_hat']:.3f}")
print(f"Effective sample size: {diagnostics['n_eff']:.0f}")
print(f"Monte Carlo SE: {diagnostics['mcse']:.4f}")

# Posterior predictive checks
ppc_results = model.posterior_predictive_check()
print(f"Posterior predictive p-value: {ppc_results['pp_p_value']:.3f}")
''',

        'power_analysis': '''
# Statistical Power Analysis
from modeling.statistical_analysis import StatisticalModel, StatisticalConfig, TestType

# Configure power analysis
config = StatisticalConfig(
    test_type=TestType.INDEPENDENT_T_TEST,
    alpha_level=0.05,
    power_target=0.80,
    effect_size=0.5,  # Medium effect size
    alternative='two-sided'
)

# Prospective power analysis (find required sample size)
model = StatisticalModel(config, name="Power_Analysis")
power_results = model.calculate_power(analysis_type='prospective')

print(f"Required sample size per group: {power_results.n_per_group}")
print(f"Total sample size needed: {power_results.total_n}")
print(f"Achieved power: {power_results.achieved_power:.3f}")

# Retrospective power analysis (given sample size)
retrospective_power = model.calculate_power(
    analysis_type='retrospective',
    n_per_group=25
)
print(f"Power with n=25 per group: {retrospective_power.power:.3f}")

# Sensitivity analysis (minimum detectable effect)
sensitivity = model.calculate_power(
    analysis_type='sensitivity',
    n_per_group=30,
    power=0.80
)
print(f"Minimum detectable effect size: {sensitivity.min_effect_size:.3f}")

# Generate power curve
power_curve = model.generate_power_curve(
    effect_sizes=np.arange(0.1, 1.5, 0.1),
    sample_sizes=[10, 20, 30, 50, 100]
)
# power_curve contains data for plotting power vs effect size
'''
    }


def get_statistical_guidelines() -> Dict[str, Any]:
    """Get comprehensive statistical analysis guidelines and best practices."""
    return {
        'study_design_guidelines': {
            'sample_size_determination': {
                'power_analysis': 'Conduct a priori power analysis for adequate sample size',
                'effect_size_estimation': 'Use pilot data or literature to estimate expected effect sizes',
                'multiple_comparisons': 'Account for multiple comparisons in sample size calculations',
                'attrition_planning': 'Plan for potential data loss and increase sample accordingly'
            },
            'experimental_design': {
                'randomization': 'Use proper randomization procedures to ensure group equivalence',
                'blinding': 'Implement blinding when possible to reduce bias',
                'control_groups': 'Include appropriate control or comparison groups',
                'blocking_stratification': 'Use blocking or stratification for known confounders'
            },
            'data_collection': {
                'measurement_quality': 'Ensure reliable and valid measurement instruments',
                'missing_data_prevention': 'Implement strategies to minimize missing data',
                'outlier_documentation': 'Document reasons for any extreme or unusual observations',
                'protocol_adherence': 'Maintain strict adherence to data collection protocols'
            }
        },
        'analysis_selection_guidelines': {
            'test_selection_flowchart': {
                'data_type_assessment': 'Identify whether data is continuous, ordinal, or categorical',
                'distribution_assessment': 'Check normality and other distributional assumptions',
                'sample_size_consideration': 'Consider sample size limitations for test selection',
                'research_question_alignment': 'Ensure test matches the specific research question'
            },
            'assumption_checking_priority': {
                'independence': 'Always verify independence of observations first',
                'normality': 'Test normality for parametric procedures',
                'homogeneity': 'Check equal variances for group comparison tests',
                'linearity': 'Assess linearity for regression analyses'
            },
            'robustness_considerations': {
                'assumption_violations': 'Use robust alternatives when assumptions are violated',
                'sensitivity_analysis': 'Conduct sensitivity analyses with different methods',
                'bootstrap_validation': 'Use bootstrap methods for robust inference',
                'cross_validation': 'Validate findings with independent data when possible'
            }
        },
        'interpretation_guidelines': {
            'significance_interpretation': {
                'p_value_meaning': 'Interpret p-values as evidence against null hypothesis',
                'clinical_significance': 'Consider practical significance beyond statistical significance',
                'confidence_intervals': 'Report and interpret confidence intervals for effect estimates',
                'effect_size_reporting': 'Always report appropriate effect size measures'
            },
            'effect_size_interpretation': {
                'cohens_guidelines': 'Use Cohen\'s conventions as rough guidelines (small: 0.2, medium: 0.5, large: 0.8)',
                'context_specific': 'Consider field-specific benchmarks for effect size interpretation',
                'confidence_intervals': 'Report confidence intervals for effect sizes',
                'practical_importance': 'Evaluate practical importance of observed effects'
            },
            'multiple_testing': {
                'correction_necessity': 'Apply appropriate corrections for multiple comparisons',
                'planned_vs_exploratory': 'Distinguish between planned and exploratory analyses',
                'family_definition': 'Clearly define the family of tests for correction',
                'alpha_adjustment': 'Document and justify alpha level adjustments'
            }
        },
        'reporting_guidelines': {
            'apa_style_reporting': {
                'descriptive_statistics': 'Report means, standard deviations, and sample sizes',
                'test_statistics': 'Include test statistic, degrees of freedom, and exact p-values',
                'effect_sizes': 'Report appropriate effect size measures with confidence intervals',
                'assumption_testing': 'Report results of assumption tests when relevant'
            },
            'graphical_presentation': {
                'data_visualization': 'Use appropriate plots to display data distributions',
                'error_bars': 'Include appropriate error bars (SE, SD, or CI)',
                'effect_plots': 'Show effect sizes and their confidence intervals',
                'diagnostic_plots': 'Include relevant diagnostic plots for model checking'
            },
            'reproducibility': {
                'analysis_code': 'Provide complete analysis code for reproducibility',
                'software_versions': 'Document software and package versions used',
                'random_seeds': 'Set and report random seeds for reproducible results',
                'data_availability': 'Make data available when possible and appropriate'
            }
        },
        'common_pitfalls': {
            'design_pitfalls': {
                'post_hoc_power': 'Avoid post-hoc power analyses on non-significant results',
                'data_dredging': 'Avoid excessive data exploration without hypothesis correction',
                'pseudoreplication': 'Ensure proper identification of independent units',
                'survivor_bias': 'Account for potential selection or survival biases'
            },
            'analysis_pitfalls': {
                'assumption_ignorance': 'Never ignore assumption violations',
                'outlier_deletion': 'Avoid arbitrary deletion of outliers without justification',
                'model_overfitting': 'Be cautious of overfitting, especially with small samples',
                'correlation_causation': 'Remember correlation does not imply causation'
            },
            'interpretation_pitfalls': {
                'p_hacking': 'Avoid selective reporting of significant results',
                'effect_size_ignorance': 'Don\'t ignore effect sizes in favor of p-values only',
                'confidence_interval_misinterpretation': 'Understand proper CI interpretation',
                'base_rate_neglect': 'Consider base rates when interpreting test results'
            }
        }
    }


def get_performance_benchmarks() -> Dict[str, Any]:
    """Get performance benchmarks and computational expectations."""
    return {
        'computational_performance': {
            'small_datasets': {
                'description': '< 1,000 observations, < 20 variables',
                'typical_computation_time': '< 1 second for most analyses',
                'memory_usage': '< 100 MB RAM',
                'recommended_methods': 'All methods suitable',
                'special_considerations': 'Be cautious with complex models and small samples'
            },
            'medium_datasets': {
                'description': '1,000 - 100,000 observations, 20-100 variables',
                'typical_computation_time': '1-30 seconds for most analyses',
                'memory_usage': '100 MB - 2 GB RAM',
                'recommended_methods': 'All methods, consider robust alternatives',
                'special_considerations': 'Assumption checking becomes more reliable'
            },
            'large_datasets': {
                'description': '> 100,000 observations, > 100 variables',
                'typical_computation_time': '30 seconds - 10 minutes',
                'memory_usage': '2-16 GB RAM',
                'recommended_methods': 'Robust methods preferred, consider sampling',
                'special_considerations': 'Statistical significance may not imply practical significance'
            }
        },
        'statistical_power_expectations': {
            'typical_power_levels': {
                'low_power': '< 0.50 - Often inadequate for reliable conclusions',
                'moderate_power': '0.50 - 0.80 - Acceptable for exploratory studies',
                'adequate_power': '0.80 - 0.90 - Standard for confirmatory studies',
                'high_power': '> 0.90 - Excellent for definitive studies'
            },
            'sample_size_guidelines': {
                't_tests': 'n ≥ 30 per group for moderate effects with 80% power',
                'anova': 'n ≥ 20 per group for 3-4 groups with moderate effects',
                'regression': 'n ≥ 10-20 per predictor variable',
                'correlation': 'n ≥ 85 for detecting r = 0.3 with 80% power'
            }
        },
        'accuracy_expectations': {
            'type_i_error_control': {
                'nominal_alpha': '0.05 for most applications',
                'achieved_alpha': 'Should be ≤ nominal level under null hypothesis',
                'multiple_testing': 'Family-wise error rate control when appropriate',
                'false_discovery_rate': 'FDR control for exploratory analyses'
            },
            'effect_size_detection': {
                'small_effects': 'Cohen\'s d = 0.2, requires large samples (n > 400 per group)',
                'medium_effects': 'Cohen\'s d = 0.5, moderate samples (n ≈ 60 per group)',
                'large_effects': 'Cohen\'s d = 0.8, small samples sufficient (n ≈ 25 per group)',
                'practical_significance': 'Consider domain-specific meaningful effect sizes'
            }
        },
        'software_performance': {
            'computation_optimization': {
                'vectorized_operations': 'Efficient numpy/scipy implementations',
                'parallel_processing': 'Multi-core support for computationally intensive methods',
                'memory_management': 'Efficient memory usage for large datasets',
                'numerical_stability': 'Robust numerical algorithms for edge cases'
            },
            'algorithm_complexity': {
                'basic_tests': 'O(n) - Linear time complexity',
                'bootstrap_methods': 'O(n × B) - Linear in sample size and bootstrap samples',
                'permutation_tests': 'O(n × P) - Linear in sample size and permutations',
                'mcmc_methods': 'O(n × I × C) - Linear in data, iterations, and chains'
            }
        }
    }


def generate_info_summary() -> str:
    """Generate a comprehensive summary of the statistical analysis module."""
    info = get_package_info()
    methods = get_method_comparison()
    
    summary = f"""
# Statistical Analysis Module Summary

## Overview
{info['description']}

**Version:** {info['version']}
**Last Updated:** {info['last_updated']}

## Key Capabilities
- **12+ Statistical Methods** across parametric, non-parametric, and Bayesian approaches
- **Comprehensive Assumption Testing** with automated diagnostics and validation
- **Advanced Effect Size Calculations** with multiple standardized measures
- **Power Analysis Framework** for prospective, retrospective, and sensitivity analysis
- **Robust Statistical Inference** with bootstrap and permutation methods
- **Multiple Comparison Corrections** (Bonferroni, FDR, Holm-Bonferroni)

## Supported Method Categories
- **Hypothesis Testing:** {', '.join(['T-tests', 'ANOVA', 'Chi-square', 'Non-parametric'])}
- **Regression Analysis:** {', '.join(['Linear', 'Logistic', 'Robust', 'Time Series'])}
- **Bayesian Methods:** {', '.join(['Bayesian Inference', 'MCMC Sampling'])}
- **Multivariate Analysis:** {', '.join(['MANOVA', 'Factor Analysis', 'Canonical Correlation'])}

## Problem Types Supported
- Hypothesis Testing and Group Comparisons
- Relationship and Correlation Analysis  
- Distribution Analysis and Goodness-of-Fit
- Time Series Statistical Analysis
- Multivariate Statistical Modeling

## Advanced Features
- **Power Analysis:** Sample size calculation and power curves
- **Effect Sizes:** Cohen's d, η², ω², Cramér's V, and more
- **Robust Methods:** Outlier-resistant statistical procedures
- **Bayesian Framework:** MCMC sampling with comprehensive diagnostics
- **Assumption Testing:** Automated validation of statistical assumptions

## Integration
- ✅ SciPy/StatsModels Backend
- ✅ Scikit-learn Compatible  
- ✅ Pandas DataFrame Support
- ✅ BaseModel Interface Compliance
- ✅ R Integration (Optional)
- ✅ Interactive Visualization

## Quick Start
```python
from modeling.statistical_analysis import StatisticalModel, StatisticalConfig, TestType

config = StatisticalConfig(test_type=TestType.INDEPENDENT_T_TEST)
model = StatisticalModel(config)
results = model.fit(group1_data, group2_data)
print(f"p-value: {{results.p_value:.4f}}, Effect size: {{results.effect_size:.3f}}")
```

For detailed usage examples and advanced statistical procedures, see the full documentation.
"""
    return summary.strip()


def export_info_json(filename: str = 'statistical_analysis_info.json') -> None:
    """Export complete module information to JSON file."""
    info_data = {
        'package_info': get_package_info(),
        'method_comparison': get_method_comparison(),
        'implementation_examples': get_implementation_examples(),
        'statistical_guidelines': get_statistical_guidelines(),
        'performance_benchmarks': get_performance_benchmarks(),
        'generated_at': datetime.now().isoformat()
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Statistical analysis module information exported to {filename}")
        print(f"✅ Module information exported to {filename}")
    except Exception as e:
        logger.error(f"Failed to export module information: {e}")
        print(f"❌ Export failed: {e}")


if __name__ == "__main__":
    # Generate and display summary when run directly
    print("📊 Statistical Analysis Module Information")
    print("=" * 60)
    print(generate_info_summary())
    print("\n" + "=" * 60)
    
    # Optionally export to JSON
    export_choice = input("\nExport detailed information to JSON? (y/n): ").lower().strip()
    if export_choice in ['y', 'yes']:
        export_info_json()
    
    print("\n📚 Documentation complete!")
