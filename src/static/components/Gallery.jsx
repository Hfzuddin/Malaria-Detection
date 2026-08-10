const { useState, useEffect } = React;

function Gallery() {
  const [metrics, setMetrics] = useState({ accuracy: '95.76%', precision: '95.89%', recall: '95.89%', f1: '95.89%' });
  const [showImage, setShowImage] = useState(false);

  useEffect(() => {
    fetch('/api/model_metrics')
      .then(res => res.json())
      .then(data => {
        if (data.accuracy && data.accuracy !== 'N/A') {
          setMetrics({
            accuracy: (data.accuracy * 100).toFixed(2) + '%',
            precision: (data.precision * 100).toFixed(2) + '%',
            recall: (data.recall * 100).toFixed(2) + '%',
            f1: (data.f1 * 100).toFixed(2) + '%'
          });
        }
      }).catch(console.error);
  }, []);

  return (
    <div>
      <h2 className="text-2xl font-semibold leading-tight text-gray-800 dark:text-white transition-colors mb-6">
        Model Analytics
      </h2>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4 mb-8">
        {[
          { name: 'Accuracy', stat: metrics.accuracy },
          { name: 'Precision', stat: metrics.precision },
          { name: 'Recall', stat: metrics.recall },
          { name: 'F1-Score', stat: metrics.f1 },
        ].map((item) => (
          <div key={item.name} className="overflow-hidden rounded-lg bg-white dark:bg-zinc-900 px-4 py-5 shadow border border-gray-200 dark:border-zinc-800 sm:p-6 transition-colors duration-200">
            <dt className="truncate text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">{item.name}</dt>
            <dd className="mt-1 text-3xl font-semibold tracking-tight text-gray-900 dark:text-gray-100">{item.stat}</dd>
          </div>
        ))}
      </div>

      <div className="bg-white dark:bg-zinc-900 shadow sm:rounded-lg border border-gray-200 dark:border-zinc-800 mb-8 overflow-hidden transition-colors duration-200">
        <div className="px-4 py-5 sm:px-6 border-b border-gray-200 dark:border-zinc-800 flex justify-between items-center">
          <h3 className="text-lg font-medium leading-6 text-gray-900 dark:text-gray-100">Confusion Matrix (3-Class)</h3>
          <button onClick={() => setShowImage(!showImage)} className="text-sm bg-indigo-50 text-indigo-700 hover:bg-indigo-100 dark:bg-indigo-900/30 dark:text-indigo-400 dark:hover:bg-indigo-900/50 px-3 py-1.5 rounded-md font-medium transition-colors">
            {showImage ? 'Show Table' : 'View Image'}
          </button>
        </div>
        <div className="p-6 overflow-x-auto flex justify-center">
          {showImage ? (
            <img src="/results_images/CM_model_70_128D_30_3Class.png" alt="Confusion Matrix" className="max-w-full h-auto rounded shadow-sm border border-gray-200 dark:border-zinc-700 max-h-[600px]" />
          ) : (
            <div className="min-w-[600px] grid grid-cols-4 gap-2 text-center text-sm w-full">
            <div className="p-4"></div>
            <div className="p-4 font-semibold text-gray-600 dark:text-gray-300 border-b border-gray-200 dark:border-zinc-800">Predicted<br/>Non-bloodCell</div>
            <div className="p-4 font-semibold text-gray-600 dark:text-gray-300 border-b border-gray-200 dark:border-zinc-800">Predicted<br/>Parasitized</div>
            <div className="p-4 font-semibold text-gray-600 dark:text-gray-300 border-b border-gray-200 dark:border-zinc-800">Predicted<br/>Uninfected</div>

            <div className="p-4 font-semibold text-gray-600 dark:text-gray-300 border-r border-gray-200 dark:border-zinc-800 flex items-center justify-end pr-6">Actual<br/>Non-bloodCell</div>
            <div className="p-4 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 rounded-md border border-green-200 dark:border-green-800/30 flex flex-col justify-center"><span className="text-xl font-bold">1249</span><span className="text-xs uppercase mt-1">True Positive</span></div>
            <div className="p-4 bg-gray-50 dark:bg-zinc-800 text-gray-500 dark:text-gray-400 rounded-md flex flex-col justify-center"><span className="text-xl font-bold">0</span><span className="text-xs uppercase mt-1">False Negative</span></div>
            <div className="p-4 bg-gray-50 dark:bg-zinc-800 text-gray-500 dark:text-gray-400 rounded-md flex flex-col justify-center"><span className="text-xl font-bold">0</span><span className="text-xs uppercase mt-1">False Negative</span></div>

            <div className="p-4 font-semibold text-gray-600 dark:text-gray-300 border-r border-gray-200 dark:border-zinc-800 flex items-center justify-end pr-6">Actual<br/>Parasitized</div>
            <div className="p-4 bg-gray-50 dark:bg-zinc-800 text-gray-500 dark:text-gray-400 rounded-md flex flex-col justify-center"><span className="text-xl font-bold">0</span><span className="text-xs uppercase mt-1">False Negative</span></div>
            <div className="p-4 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 rounded-md border border-green-200 dark:border-green-800/30 flex flex-col justify-center"><span className="text-xl font-bold">1284</span><span className="text-xs uppercase mt-1">True Positive</span></div>
            <div className="p-4 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 rounded-md border border-red-200 dark:border-red-800/30 flex flex-col justify-center"><span className="text-xl font-bold">94</span><span className="text-xs uppercase mt-1">False Negative</span></div>

            <div className="p-4 font-semibold text-gray-600 dark:text-gray-300 border-r border-gray-200 dark:border-zinc-800 flex items-center justify-end pr-6">Actual<br/>Uninfected</div>
            <div className="p-4 bg-gray-50 dark:bg-zinc-800 text-gray-500 dark:text-gray-400 rounded-md flex flex-col justify-center"><span className="text-xl font-bold">0</span><span className="text-xs uppercase mt-1">False Positive</span></div>
            <div className="p-4 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 rounded-md border border-red-200 dark:border-red-800/30 flex flex-col justify-center"><span className="text-xl font-bold">76</span><span className="text-xs uppercase mt-1">False Positive</span></div>
            <div className="p-4 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 rounded-md border border-green-200 dark:border-green-800/30 flex flex-col justify-center"><span className="text-xl font-bold">1302</span><span className="text-xs uppercase mt-1">True Positive</span></div>
          </div>
          )}
        </div>
      </div>
      
      <div className="bg-white dark:bg-zinc-900 shadow sm:rounded-lg border border-gray-200 dark:border-zinc-800 p-6 transition-colors duration-200">
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">About the Model</h3>
        <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed">
          The system uses a custom Hybrid EfficientNet architecture optimized for extracting features from blood smear images. 
          These features are then classified using a Support Vector Machine (SVM) to detect malaria parasitized cells.
        </p>
      </div>
    </div>
  );
}
