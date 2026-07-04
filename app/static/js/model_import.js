/**
 * 模型导入 JS — 支持拖拽上传、AI 推荐预览、一键确认
 */

// ── 状态 ──────────────────────────────────────────
var importState = {
    file: null,            // 选中的 File 对象
    extracted: null,       // /import/preview 返回的 extracted
    recommendations: null, // /import/preview 返回的 recommendations
    originalName: '',      // 推荐名称 (供 "应用推荐" 恢复)
    originalDesc: '',
    originalVersion: '',
};

var UPLOAD_URL = '/api/v1/models/import/preview';
var CONFIRM_URL = '/api/v1/models/import/confirm';

// ── DOM 缓存 ──────────────────────────────────────
var $ = function(id) { return document.getElementById(id); };

// ── 初始化 ─────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
    setupDropZone();
    setupFileInput();
    setupFormSubmit();
});

// ── 拖拽区 ─────────────────────────────────────────
function setupDropZone() {
    var dz = $('dropZone');
    var input = $('fileInput');

    dz.addEventListener('click', function(e) {
        if (e.target.tagName !== 'BUTTON' && e.target.tagName !== 'INPUT') {
            input.click();
        }
    });

    dz.addEventListener('dragover', function(e) {
        e.preventDefault();
        dz.style.borderColor = 'var(--bs-primary)';
        dz.style.backgroundColor = 'rgba(13,110,253,0.05)';
    });

    dz.addEventListener('dragleave', function() {
        dz.style.borderColor = '#dee2e6';
        dz.style.backgroundColor = '';
    });

    dz.addEventListener('drop', function(e) {
        e.preventDefault();
        dz.style.borderColor = '#dee2e6';
        dz.style.backgroundColor = '';
        var files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    input.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFileSelect(this.files[0]);
        }
    });
}

function setupFileInput() {
    // 已通过 setupDropZone 中的 input change 处理
}

// ── 文件选择 ───────────────────────────────────────
function handleFileSelect(file) {
    // 校验扩展名
    var ext = file.name.split('.').pop().toLowerCase();
    var allowed = ['pkl', 'pt', 'pth', 'h5', 'keras', 'joblib', 'zip'];
    if (allowed.indexOf(ext) === -1) {
        showError('不支持的文件格式: .' + ext, '请选择 .pkl, .pt, .h5, .keras 或 .zip 文件。');
        return;
    }

    importState.file = file;
    updateFileDisplay(file);
    uploadAndPreview(file);
}

function updateFileDisplay(file) {
    $('uploadCard').querySelector('.card-header h6').textContent = '1. 已选择文件';
    $('dropZoneContent').style.display = 'none';
    $('fileSelected').style.display = 'block';
    $('fileNameDisplay').textContent = file.name;
    $('fileSizeDisplay').textContent = formatSize(file.size);
}

function resetUpload() {
    importState.file = null;
    importState.extracted = null;
    importState.recommendations = null;
    $('uploadCard').querySelector('.card-header h6').textContent = '1. 选择模型文件';
    $('dropZoneContent').style.display = 'block';
    $('fileSelected').style.display = 'none';
    $('previewCard').style.display = 'none';
    $('loadingCard').style.display = 'none';
    $('successCard').style.display = 'none';
    $('errorCard').style.display = 'none';
    $('fileInput').value = '';
}

function resetAll() {
    resetUpload();
}

// ── 上传 + 预览 ────────────────────────────────────
function uploadAndPreview(file) {
    $('loadingCard').style.display = 'block';
    $('previewCard').style.display = 'none';
    $('errorCard').style.display = 'none';

    var formData = new FormData();
    formData.append('model_file', file);

    fetch(UPLOAD_URL, {
        method: 'POST',
        headers: {
            'X-CSRF-Token': getCsrfToken(),
        },
        body: formData,
    })
    .then(function(r) { return r.json(); })
    .then(function(resp) {
        $('loadingCard').style.display = 'none';
        if (resp.success) {
            importState.extracted = resp.data.extracted;
            importState.recommendations = resp.data.recommendations;
            showPreview(resp.data);
        } else {
            showError('解析失败', resp.message || '无法解析模型文件。');
        }
    })
    .catch(function(err) {
        $('loadingCard').style.display = 'none';
        showError('网络错误', '请求失败: ' + err.message);
    });
}

// ── 展示预览 ───────────────────────────────────────
function showPreview(data) {
    var ext = data.extracted;
    var rec = data.recommendations;

    // 保存原始推荐值
    importState.originalName = rec.name || '';
    importState.originalDesc = rec.description || '';
    importState.originalVersion = rec.version || '1.0.0';

    // 填充表单
    $('fieldName').value = rec.name || '';
    $('fieldDescription').value = rec.description || '';
    $('fieldVersion').value = rec.version || '1.0.0';
    $('fieldModelType').value = ext.model_type || 'other';
    $('fieldFramework').value = ext.framework || (ext.has_metadata_json ? 'auto' : 'auto');

    // 提取的信息摘要
    $('infoAlgorithm').textContent = ext.algorithm || '-';
    $('infoFeatures').textContent = ext.feature_names
        ? ext.feature_names.length + ' 个'
        : (ext.input_dimension ? ext.input_dimension + ' 维' : '-');
    $('infoClasses').textContent = ext.class_labels && ext.class_labels.length > 0
        ? ext.class_labels.slice(0, 5).join(', ') + (ext.class_labels.length > 5 ? '...' : '')
        : '-';

    if (ext.has_metadata_json) {
        $('infoMetaSource').textContent = 'metadata.json';
        $('metadataBadge').textContent = '完整元数据';
        $('metadataBadge').className = 'badge bg-success';
        // 显示已有指标
        var meta = ext.existing_metadata || {};
        var metrics = [];
        if (meta.accuracy !== null && meta.accuracy !== undefined) metrics.push('Accuracy: ' + (meta.accuracy * 100).toFixed(1) + '%');
        if (meta.f1_score !== null && meta.f1_score !== undefined) metrics.push('F1: ' + meta.f1_score.toFixed(4));
        if (meta.r2 !== null && meta.r2 !== undefined) metrics.push('R²: ' + meta.r2.toFixed(4));
        if (meta.mse !== null && meta.mse !== undefined) metrics.push('MSE: ' + meta.mse.toFixed(4));
        if (metrics.length > 0) {
            $('infoMetrics').style.display = 'block';
            $('infoMetricsValues').textContent = metrics.join(' | ');
        }
    } else {
        $('infoMetaSource').textContent = '模型文件解析';
        $('metadataBadge').textContent = 'AI 推荐';
        $('metadataBadge').className = 'badge bg-info';
    }

    $('previewCard').style.display = 'block';
    $('errorCard').style.display = 'none';

    // 滚动到预览区
    $('previewCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── 应用 AI 推荐 ───────────────────────────────────
function applyRecommendation(field) {
    if (field === 'name' && importState.originalName) {
        $('fieldName').value = importState.originalName;
    } else if (field === 'description' && importState.originalDesc) {
        $('fieldDescription').value = importState.originalDesc;
    } else if (field === 'version' && importState.originalVersion) {
        $('fieldVersion').value = importState.originalVersion;
    }
}

// ── 提交确认 ───────────────────────────────────────
function setupFormSubmit() {
    $('importForm').addEventListener('submit', function(e) {
        e.preventDefault();
        confirmImport();
    });
}

function confirmImport() {
    var name = $('fieldName').value.trim();
    if (!name) {
        showToast('请填写模型名称', 'danger');
        $('fieldName').focus();
        return;
    }

    var btn = $('confirmBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> 导入中...';

    var formData = new FormData();
    formData.append('model_file', importState.file);
    formData.append('name', name);
    formData.append('description', $('fieldDescription').value.trim());
    formData.append('version', $('fieldVersion').value.trim() || '1.0.0');
    formData.append('model_type', $('fieldModelType').value);
    formData.append('is_public', $('fieldIsPublic').checked ? 'true' : 'false');

    // 如果有 metadata.json, 回传指标
    var ext = importState.extracted;
    if (ext && ext.has_metadata_json && ext.existing_metadata) {
        var meta = ext.existing_metadata;
        var metrics = {};
        var metricKeys = ['accuracy','precision','recall','f1_score','loss','r2','mse','mae'];
        metricKeys.forEach(function(k) {
            if (meta[k] !== null && meta[k] !== undefined) {
                metrics[k] = meta[k];
            }
        });
        if (Object.keys(metrics).length > 0) {
            formData.append('metrics', JSON.stringify(metrics));
        }
        // 回传超参数
        if (meta.hyperparameters && Object.keys(meta.hyperparameters).length > 0) {
            formData.append('hyperparameters', JSON.stringify(meta.hyperparameters));
        }
    }

    fetch(CONFIRM_URL, {
        method: 'POST',
        headers: {
            'X-CSRF-Token': getCsrfToken(),
        },
        body: formData,
    })
    .then(function(r) { return r.json(); })
    .then(function(resp) {
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-check-circle"></i> 确认导入';
        if (resp.success) {
            showSuccess(resp.data);
        } else {
            showToast(resp.message || '导入失败', 'danger');
        }
    })
    .catch(function(err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-check-circle"></i> 确认导入';
        showToast('网络错误: ' + err.message, 'danger');
    });
}

// ── 成功/错误显示 ──────────────────────────────────
function showSuccess(data) {
    $('previewCard').style.display = 'none';
    $('successCard').style.display = 'block';
    $('successMessage').textContent = '模型 "' + (data.name || '') + '" 已成功导入。';
    $('successLink').href = '/models/' + data.id;
}

function showError(title, message) {
    $('loadingCard').style.display = 'none';
    $('previewCard').style.display = 'none';
    $('errorCard').style.display = 'block';
    $('errorTitle').textContent = title || '导入失败';
    $('errorMessage').textContent = message || '';
}

// ── 工具函数 ───────────────────────────────────────
function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(2) + ' MB';
}

function getCsrfToken() {
    var meta = document.querySelector('meta[name="csrf-token"]');
    if (meta) return meta.getAttribute('content');
    // Fallback: 从 cookie 或 header 获取
    return '';
}

function showToast(msg, type) {
    // 使用全局通知函数 (如果存在)
    if (typeof showCenteredMessage === 'function') {
        showCenteredMessage(msg, type || 'info', 3000);
        return;
    }
    // 降级: alert
    alert(msg);
}
