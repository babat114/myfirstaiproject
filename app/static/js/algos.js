/* ============================================
   Algorithm config — shared across training templates
   ============================================ */
const ALGORITHMS = {
    classification: [
        {value: 'auto', label: '🤖 AutoML 智能选择 (自动遍历所有算法)'},
        {value: 'transformer_bert', label: 'BERT Transformer (NLP迁移学习)'},
        {value: 'random_forest', label: '随机森林 (Random Forest)'},
        {value: 'gradient_boosting', label: '梯度提升 (Gradient Boosting)'},
        {value: 'logistic_regression', label: '逻辑回归 (Logistic Regression)'},
        {value: 'svm', label: '支持向量机 (SVM)'},
        {value: 'knn', label: 'K近邻 (KNN)'},
        {value: 'decision_tree', label: '决策树 (Decision Tree)'},
        {value: 'mlp', label: 'PyTorch MLP (深度学习)'},
    ],
    regression: [
        {value: 'auto', label: '🤖 AutoML 智能选择 (自动遍历所有算法)'},
        {value: 'random_forest_regressor', label: '随机森林回归 (Random Forest)'},
        {value: 'gradient_boosting_regressor', label: '梯度提升回归 (Gradient Boosting)'},
        {value: 'ridge', label: '岭回归 (Ridge, L2正则)'},
        {value: 'knn_regressor', label: 'K近邻回归 (KNN Regressor)'},
        {value: 'linear_regression', label: '线性回归 (Linear Regression)'},
        {value: 'svr', label: '支持向量回归 (SVR)'},
        {value: 'mlp', label: 'PyTorch MLP (深度学习)'},
    ],
    clustering: [
        {value: 'auto', label: '🤖 AutoML 智能选择 (自动遍历所有算法)'},
        {value: 'kmeans', label: 'K-Means 聚类'},
        {value: 'dbscan', label: 'DBSCAN 密度聚类'},
        {value: 'agglomerative', label: '层次聚类 (Agglomerative)'},
        {value: 'minibatch_kmeans', label: 'MiniBatch K-Means'},
    ]
};

const sklearnAlgos = ['random_forest', 'gradient_boosting', 'random_forest_regressor',
    'gradient_boosting_regressor', 'svm', 'svr', 'logistic_regression', 'linear_regression', 'ridge', 'knn',
    'knn_regressor', 'decision_tree',
    'kmeans', 'dbscan', 'agglomerative', 'minibatch_kmeans'];

const dlAlgos = ['transformer_bert', 'mlp'];

const ALGO_LABELS = {};
Object.values(ALGORITHMS).forEach(arr => arr.forEach(a => { ALGO_LABELS[a.value] = a.label; }));
