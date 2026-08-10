const { useState, useEffect } = React;

function ScannedResults() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/results')
      .then(res => res.json())
      .then(data => {
        setResults(data || []);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  const statusColors = {
    UNINFECTED: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 border-transparent',
    INVALID: 'bg-gray-100 text-gray-800 dark:bg-gray-800/80 dark:text-gray-400 border-transparent',
    INFECTED: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400 border-transparent',
  };

  if (loading) return <div className="text-center p-10 text-gray-500 dark:text-gray-400">Loading results...</div>;

  return (
    <div>
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
        <h2 className="text-2xl font-semibold leading-tight text-gray-800 dark:text-white transition-colors">
          Scanned Results
        </h2>
      </div>

      <div className="overflow-hidden bg-white dark:bg-zinc-900 shadow sm:rounded-lg border border-gray-200 dark:border-zinc-800 transition-colors duration-200">
        <div className="p-0">
          {results.length === 0 ? (
            <div className="text-center py-16">
              <p className="text-gray-500 dark:text-gray-400 text-lg">No scans tracked yet.</p>
              <p className="text-sm text-gray-400 dark:text-gray-500 mt-2">Go to Upload Scan to process some images.</p>
            </div>
          ) : (
            <div className="overflow-x-auto overflow-y-auto max-h-[600px]">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-zinc-800 text-left">
                <thead className="bg-gray-50 dark:bg-zinc-900/50">
                  <tr className="divide-x divide-gray-200 dark:divide-zinc-800">
                    <th scope="col" className="px-6 py-4 text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider w-16 text-center">No.</th>
                    <th scope="col" className="px-6 py-4 text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Image File</th>
                    <th scope="col" className="px-6 py-4 text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Diagnosis</th>
                    <th scope="col" className="px-6 py-4 text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Confidence</th>
                    <th scope="col" className="px-6 py-4 text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Processing Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-zinc-800 bg-white dark:bg-zinc-900">
                  {results.map((res, index) => {
                    const diagnosisStr = (res.diagnosis || '').toUpperCase();
                    let sKey = 'INVALID';
                    if (diagnosisStr.includes('INFECTED') && !diagnosisStr.includes('UNINFECTED')) sKey = 'INFECTED';
                    else if (diagnosisStr === 'UNINFECTED') sKey = 'UNINFECTED';

                    return (
                      <tr key={index} className="divide-x divide-gray-200 dark:divide-zinc-800 hover:bg-gray-50 dark:hover:bg-zinc-800/50 transition duration-150">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-500 dark:text-gray-400 text-center">
                          {index + 1}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center gap-4">
                            {res.original_image && (
                              <img src={`/processed_images/${res.original_image}`} alt="Cell" className="h-10 w-10 rounded-md object-cover border border-gray-200 dark:border-zinc-700" />
                            )}
                            <span className="text-sm font-medium text-gray-900 dark:text-gray-100">{res.filename}</span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full capitalize border ${statusColors[sKey]}`}>
                            {res.diagnosis || 'INVALID'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600 dark:text-gray-400">
                          {res.confidence ? res.confidence + '%' : '0%'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600 dark:text-gray-400">
                          {res.processing_time ? res.processing_time + 's' : '-'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
