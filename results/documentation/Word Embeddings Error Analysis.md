## Error Analysis

### Quantitative Analysis

The GloVe-based Gaussian Naive Bayes classifier achieved an accuracy of 69.66%, compared with 84.32% for the majority-class baseline. The confusion matrix shows 493 false positives and 2,102 false negatives, with false negatives accounting for approximately 81.0% of all misclassifications. This indicates that the model frequently misclassifies positive reviews as negative.

Although the classifier achieves a positive-class precision of 91.20%, its recall is only 70.85%. The results suggest that the model has difficulty maintaining reliable sentiment classification performance on this imbalanced dataset.

### Qualitative Analysis

Examination of ten misclassified reviews reveals several potential error patterns, including mixed sentiment, contextual ambiguity, and product-specific complaints.

For example, a positive review describing a product as "very good" while expressing concern about trans fat was incorrectly classified as negative. In contrast, a negative review complaining about an incorrect product quantity was classified as positive despite the customer's overall dissatisfaction.

These examples suggest that the model may struggle to distinguish the overall sentiment of a review from individual positive or negative expressions.

### Word Contribution Analysis

The word contribution analysis provides further evidence of potential weaknesses in the learned text representation. Product-related words such as "tea" (+2.63) and "coffee" (+2.15) show relatively strong positive contributions, despite not being inherently positive sentiment expressions.

Meanwhile, common function words such as "the" (−1.67) and "it" (−1.41) show negative contributions.

These observations suggest that the classifier may rely partly on topic-related or incidental lexical patterns rather than sentiment-specific information. However, additional experiments would be required to establish whether these patterns directly cause individual prediction errors.

### Conclusion

Overall, the results suggest that the GloVe-based Gaussian Naive Bayes classifier has difficulty distinguishing contextual sentiment from superficial lexical patterns. Its performance may be affected by mixed sentiment expressions, class imbalance, and correlations between product vocabulary and sentiment labels.

Future improvements could investigate alternative text representations, class-balancing approaches, and contextual models such as DistilBERT.
