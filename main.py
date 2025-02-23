def run_experiments(data, labels, window_sizes=[10, 20, 30]):
    # 假设data形状为(n_samples, time_steps, n_features)
    n_samples, time_steps, n_features = data.shape

    # 遍历不同的窗口长度
    for window_size in window_sizes:
        print(f"\n=== Window Size: {window_size} ===")

        # 滑动窗口分割
        X_windows = []
        y_windows = []
        for i in range(n_samples):
            for j in range(0, time_steps - window_size + 1, window_size // 2):
                X_windows.append(data[i, j:j + window_size])
                y_windows.append(labels[i])
        X = np.array(X_windows)
        y = np.array(y_windows)

        # 1. 测试局部和全局归一化
        for norm_method in ['local', 'global']:
            X_norm = preprocess_data(X, norm_method)

            # 2. 测试过采样和降采样
            for sampling_method in ['original', 'oversample', 'undersample']:
                if sampling_method == 'oversample':
                    smote = SMOTE()
                    X_flat = X_norm.reshape(len(X_norm), -1)
                    X_resampled, y_resampled = smote.fit_resample(X_flat, y)
                    X_resampled = X_resampled.reshape(-1, window_size, n_features)
                elif sampling_method == 'undersample':
                    rus = RandomUnderSampler()
                    X_flat = X_norm.reshape(len(X_norm), -1)
                    X_resampled, y_resampled = rus.fit_resample(X_flat, y)
                    X_resampled = X_resampled.reshape(-1, window_size, n_features)
                else:
                    X_resampled, y_resampled = X_norm, y

                # 3. 测试数据增强
                for aug_method in ['none', 'noise', 'transpose', 'peaks']:
                    X_aug = X_resampled.copy()
                    if aug_method == 'noise':
                        X_aug = add_noise(X_aug)
                    elif aug_method == 'transpose':
                        X_aug = random_transpose(X_aug)
                    elif aug_method == 'peaks':
                        X_aug = add_random_peaks(X_aug)

                    # 4. 测试时间序列和统计特征输入
                    for input_type in ['timeseries', 'stats']:
                        if input_type == 'timeseries':
                            X_input = X_aug.reshape(len(X_aug), -1)
                            input_size = window_size * n_features
                        else:
                            X_input = extract_stat_features(X_aug.reshape(len(X_aug), -1))
                            input_size = 5  # 5个统计特征

                        desc = (f"Norm: {norm_method}, Sampling: {sampling_method}, "
                                f"Aug: {aug_method}, Input: {input_type}")
                        train_and_evaluate(X_input, y_resampled, input_size, window_size, desc)


# 示例数据生成和运行
if __name__ == "__main__":
    # 生成模拟电力数据
    np.random.seed(42)
    n_samples = 1000
    time_steps = 50
    n_features = 3
    data = np.random.randn(n_samples, time_steps, n_features)
    # 生成不平衡的6类标签
    labels = np.random.choice(6, n_samples, p=[0.4, 0.2, 0.15, 0.15, 0.05, 0.05])

    print("Class distribution:", Counter(labels))
    run_experiments(data, labels)