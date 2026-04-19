// Vite static import resolution for local json files
const metricsImport = import.meta.glob('../../../../results/*/*/all_samples/metrics_summary.json', { eager: true });
const compImport = import.meta.glob('../../../../results/comparison/*/all_samples/comparison_summary.json', { eager: true });
const resultsImportMap = import.meta.glob('../../../../results/*/*/all_samples/results.json');
const datasetImport = import.meta.glob('../../../../Datasets/test2_final_hardened_v2_cleaned.json', { eager: true });

// Extract available models from the paths
export const getAvailableModels = () => {
  const models = new Set();
  Object.keys(metricsImport).forEach(path => {
    // path e.g.: ../../../../results/m1/openai-gpt-oss-120b/all_samples/metrics_summary.json
    const parts = path.split('/');
    const modelSlug = parts[parts.length - 3];
    if (modelSlug) models.add(modelSlug);
  });
  return Array.from(models);
};

// Get preloaded metrics for a specific model across all mitigations
export const getMetricsForModel = (modelSlug) => {
  const data = {};
  Object.keys(metricsImport).forEach(path => {
    if (path.includes(`/${modelSlug}/`)) {
      const parts = path.split('/');
      const mit = parts[parts.length - 4]; // "none", "m1", "m2", "m3"
      if (mit && mit !== 'comparison') {
        data[mit] = metricsImport[path].default || metricsImport[path];
      }
    }
  });
  return data;
};

// Get preloaded comparison summary
export const getComparisonSummary = (modelSlug) => {
  const path = Object.keys(compImport).find(p => p.includes(`/${modelSlug}/`));
  if (path) {
    return compImport[path].default || compImport[path];
  }
  return null;
};

// Async load results data for the charts (these are large files, so loaded dynamically)
export const loadResultsData = async (modelSlug) => {
  const data = {};
  const promises = Object.keys(resultsImportMap).map(async (path) => {
    if (path.includes(`/${modelSlug}/`)) {
      const parts = path.split('/');
      const mit = parts[parts.length - 4];
      if (mit && mit !== 'comparison') {
        const module = await resultsImportMap[path]();
        data[mit] = module.default || module;
      }
    }
  });
  await Promise.all(promises);
  return data;
};

// Get preloaded dataset
export const getDataset = () => {
  const path = Object.keys(datasetImport)[0];
  if (path) {
    return datasetImport[path].default || datasetImport[path];
  }
  return [];
};
