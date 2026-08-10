const { useState, useEffect, useRef } = React;

function Dashboard() {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [previewSrc, setPreviewSrc] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);

  const handleFiles = (files) => {
    setErrorMsg('');
    const validFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
    if (validFiles.length === 0) {
      alert("Please select valid image files");
      return;
    }
    setUploadedFiles(validFiles);
    setCurrentImageIndex(0);
  };

  useEffect(() => {
    if (uploadedFiles.length > 0) {
      const file = uploadedFiles[currentImageIndex];
      const reader = new FileReader();
      reader.onload = (e) => setPreviewSrc(e.target.result);
      reader.readAsDataURL(file);
    }
  }, [uploadedFiles, currentImageIndex]);

  const resetUpload = () => {
    setUploadedFiles([]);
    setCurrentImageIndex(0);
    setPreviewSrc('');
    setErrorMsg('');
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (folderInputRef.current) folderInputRef.current.value = '';
  };

  const analyzeImages = async () => {
    if (uploadedFiles.length === 0) return;
    setIsAnalyzing(true);
    setErrorMsg('');

    const formData = new FormData();
    uploadedFiles.forEach(file => formData.append("images", file));

    try {
      const response = await fetch('/analyze', { method: 'POST', body: formData });
      const data = await response.json();

      if (data.success) {
        let errorFound = false;
        let msg = "Invalid Blood Cell Image. Please Try Again";

        data.results?.forEach(res => {
          if (res.diagnosis === "INVALID INPUT") {
            errorFound = true;
            msg = res.error_msg || msg;
          }
        });

        if (errorFound) {
          setErrorMsg(msg);
          setIsAnalyzing(false);
        } else {
          setTimeout(() => { setIsAnalyzing(false); window.navigate?.('/history') || window.location.assign('/history'); }, 500);
        }
      } else {
        setErrorMsg("Analysis failed. Please try again.");
        setIsAnalyzing(false);
      }
    } catch (err) {
      setErrorMsg("Failed to connect to server.");
      setIsAnalyzing(false);
    }
  };

  return (
    <div>
      <h2 className="text-2xl font-semibold leading-tight text-gray-800 dark:text-white transition-colors mb-6">
        Blood Smear Scans
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div onClick={() => fileInputRef.current.click()} className="cursor-pointer border-2 border-dashed border-gray-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 rounded-lg p-10 text-center hover:border-indigo-500 dark:hover:border-indigo-500 hover:bg-gray-50 dark:hover:bg-zinc-800 transition-all">
          <svg className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
          <h3 className="mt-2 text-lg font-medium text-gray-900 dark:text-gray-100">Upload Image</h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">Select a single cell image (jpg, png)</p>
          <input type="file" ref={fileInputRef} accept="image/*" className="hidden" onChange={(e) => handleFiles(e.target.files)} />
        </div>

        <div onClick={() => folderInputRef.current.click()} className="cursor-pointer border-2 border-dashed border-gray-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 rounded-lg p-10 text-center hover:border-indigo-500 dark:hover:border-indigo-500 hover:bg-gray-50 dark:hover:bg-zinc-800 transition-all">
          <svg className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" /></svg>
          <h3 className="mt-2 text-lg font-medium text-gray-900 dark:text-gray-100">Upload Folder</h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">Select multiple images from a directory</p>
          <input type="file" ref={folderInputRef} accept="image/*" multiple webkitdirectory="true" className="hidden" onChange={(e) => handleFiles(e.target.files)} />
        </div>
      </div>

      {uploadedFiles.length > 0 && (
        <div className="bg-white dark:bg-zinc-900 shadow sm:rounded-lg overflow-hidden border border-gray-200 dark:border-zinc-800">
          <div className="px-4 py-5 sm:px-6 border-b border-gray-200 dark:border-zinc-800 flex justify-between items-center">
            <h3 className="text-lg font-medium leading-6 text-gray-900 dark:text-gray-100">Image Preview</h3>
            <div className="flex gap-4 items-center">
              <span className="inline-flex items-center rounded-md bg-indigo-50 dark:bg-indigo-900/30 px-2 py-1 text-xs font-medium text-indigo-700 dark:text-indigo-400 ring-1 ring-inset ring-indigo-700/10 dark:ring-indigo-400/30">
                {currentImageIndex + 1} / {uploadedFiles.length}
              </span>
              <button onClick={resetUpload} className="text-sm font-semibold text-red-600 hover:text-red-500 dark:text-red-400 dark:hover:text-red-300">Reset</button>
            </div>
          </div>
          <div className="p-6 bg-gray-50 dark:bg-zinc-950 flex flex-col items-center justify-center min-h-[300px]">
            {previewSrc && <img className="max-w-full max-h-[40vh] object-contain rounded-md shadow-sm" src={previewSrc} alt="Preview" />}
            <div className="mt-4 px-3 py-1 bg-white dark:bg-zinc-800 border border-gray-200 dark:border-zinc-700 rounded-md text-sm text-gray-600 dark:text-gray-400 truncate max-w-md">
              {uploadedFiles[currentImageIndex]?.name}
            </div>
          </div>
          <div className="px-4 py-4 sm:px-6 border-t border-gray-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 flex justify-between items-center">
            <div className="flex gap-2">
              {uploadedFiles.length > 1 && (
                <>
                  <button disabled={currentImageIndex === 0} onClick={() => setCurrentImageIndex(c => c - 1)} className="rounded bg-white dark:bg-zinc-800 px-3 py-1.5 text-sm font-semibold text-gray-900 dark:text-gray-100 shadow-sm ring-1 ring-inset ring-gray-300 dark:ring-zinc-700 hover:bg-gray-50 dark:hover:bg-zinc-700 disabled:opacity-50">Previous</button>
                  <button disabled={currentImageIndex === uploadedFiles.length - 1} onClick={() => setCurrentImageIndex(c => c + 1)} className="rounded bg-white dark:bg-zinc-800 px-3 py-1.5 text-sm font-semibold text-gray-900 dark:text-gray-100 shadow-sm ring-1 ring-inset ring-gray-300 dark:ring-zinc-700 hover:bg-gray-50 dark:hover:bg-zinc-700 disabled:opacity-50">Next</button>
                </>
              )}
            </div>
            <button onClick={analyzeImages} disabled={isAnalyzing} className="rounded-md bg-indigo-600 px-6 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:opacity-50">
              {isAnalyzing ? 'Analyzing...' : 'Analyze Images'}
            </button>
          </div>
          {errorMsg && (
            <div className="p-4 bg-red-50 dark:bg-red-900/30 border-t border-red-200 dark:border-red-800/30">
              <p className="text-sm text-red-600 dark:text-red-400">{errorMsg}</p>
            </div>
          )}
        </div>
      )}

      {isAnalyzing && (
        <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm z-[100] flex flex-col justify-center items-center">
          <div className="spinner"></div>
          <p className="text-white text-lg font-medium mt-4">Running CNN & SVM Analysis...</p>
        </div>
      )}
    </div>
  );
}
