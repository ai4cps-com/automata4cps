def find_optimal_threshold_for_f1(self, data, search_every=1, plot=False):
    scores_unsorted = self.anomaly_score(data[:])
    sort_ind = np.argsort(scores_unsorted)
    sort_ind = sort_ind[0::search_every]
    thresholds = scores_unsorted[sort_ind]
    labels_unsorted = data[:]['label'].cpu().detach().numpy()

    f1_scores = [f1_score(labels_unsorted != 0, scores_unsorted > th) for th in thresholds]
    opt_ind = np.argmax(f1_scores)
    opt_th = thresholds[opt_ind]
    max_f1 = f1_scores[opt_ind]
    if plot:
        fig = vis.plot2d(np.arange(0, scores_unsorted.shape[0]), scores_unsorted, return_figure=True)
        fig.add_trace(vis.plot2d(np.arange(0, labels_unsorted.shape[0]), labels_unsorted))
        fig.show()
        vis.plot2d(thresholds, f1_scores, return_figure=True).show()
    return opt_th, max_f1


def get_auroc(self, data):
    scores = self.anomaly_score(data[:])
    labels = data[:]['label'].cpu().detach().numpy()
    labels = labels != 0
    return roc_auc_score(labels, scores)


    def calculate_ad_threshold(self, d, quantile=0.95):
        """
        Estimate and store the anomaly-detection threshold from reconstruction scores.

        This method computes anomaly scores for the given dataset and sets
        ``self.threshold`` to the specified empirical quantile of those scores.
        Samples with anomaly scores above this threshold can then be flagged as
        anomalous by downstream detection routines.

        Args:
            d:
                Input data used to estimate the anomaly-score distribution. It is
                passed directly to ``self.anomaly_score(...)``.
            quantile (float, optional):
                Quantile used to define the anomaly threshold. Must typically lie in
                the interval ``[0, 1]``. For example, ``0.95`` sets the threshold
                to the 95th percentile of the observed anomaly scores.
                Defaults to 0.95.

        Returns:
            None

        Side Effects:
            - Updates ``self.threshold`` with the estimated anomaly threshold.
            - If ``self.log_mlflow`` is True, logs the threshold to MLflow under
              the metric name ``"threshold"``.

        Notes:
            - Higher quantiles produce stricter thresholds and usually result in
              fewer points being labeled as anomalous.
            - This method assumes that larger anomaly scores indicate more abnormal
              behavior.
            - If ``self.anomaly_score(...)`` returns a multidimensional array, it is
              flattened before the quantile is computed.

        Example:
            >>> model.calculate_ad_threshold(train_data, quantile=0.99)
            >>> model.threshold
        """
        scores = np.asarray(self.anomaly_score(d)).reshape(-1)
        self.threshold = np.quantile(scores, quantile)

        if self.log_mlflow:
            mlflow.log_metric('threshold', float(self.threshold))

