import { apiClient } from '../api/apiClient';

// Both demo and checkpoint modes execute on the backend. No canned predictions.
export const analysisService = {
  async runSingleAnalysis(file, query = 'What is in this satellite image?') {
    const body = new FormData();
    body.append('image', file);
    body.append('query', query);
    return apiClient.post('/analyze/single', body);
  },
  async runChangeAnalysis(beforeFile, afterFile) {
    const body = new FormData();
    body.append('image_before', beforeFile);
    body.append('image_after', afterFile);
    return apiClient.post('/analyze/change', body);
  },
  async runOpticalSarAnalysis(opticalFile, sarFile) {
    const body = new FormData();
    body.append('image_optical', opticalFile);
    body.append('image_sar', sarFile);
    return apiClient.post('/analyze/optical-sar', body);
  },
  async detectInputAndModel(files, model = 'auto') {
    const body = new FormData();
    body.append('model', model);
    files.forEach((f, i) => {
      const file = f.file || f;
      body.append(`file_${i}`, file, f.name || file.name);
    });
    return apiClient.post('/detect-model', body);
  },
};
