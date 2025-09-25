'use client'

import { useState, useRef, FormEvent } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { useRouter } from 'next/navigation'

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [thumbnails, setThumbnails] = useState<string[]>([])
  const [selectedThumbnail, setSelectedThumbnail] = useState<string>('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const router = useRouter()
  const { token } = useAuth()

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setThumbnails([])
      setSelectedThumbnail('')
      if (videoRef.current) {
        videoRef.current.src = URL.createObjectURL(selectedFile)
      }
    }
  }

  const captureThumbnails = () => {
    if (!videoRef.current || !canvasRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    if (!context) return

    const captured: string[] = []
    const interval = video.duration / 5 // 5개의 썸네일 캡처
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    let currentTime = 0
    const captureFrame = () => {
      if (currentTime > video.duration || captured.length >= 5) {
        setThumbnails(captured)
        if (captured.length > 0 && !selectedThumbnail) {
          setSelectedThumbnail(captured[0])
        }
        return
      }
      video.currentTime = currentTime
    }

    video.onseeked = () => {
      context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight)
      captured.push(canvas.toDataURL('image/jpeg'))
      currentTime += interval
      captureFrame()
    }

    captureFrame()
  }

  const handleUpload = async (e: FormEvent) => {
    e.preventDefault()
    if (!file) {
      setError('업로드할 비디오 파일을 선택해주세요.')
      return
    }
    if (!title.trim()) {
      setError('영상의 제목을 입력해주세요.')
      return
    }

    // 업로드 확인창
    if (!window.confirm('영상을 업로드 하시겠습니까?')) {
      return
    }

    setIsProcessing(true)
    setError(null)

    const formData = new FormData()
    formData.append('video', file)
    formData.append('title', title)
    if (selectedThumbnail) {
      formData.append('thumbnail', selectedThumbnail)
    }

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/embed`,
        {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${token}`,
          },
          body: formData,
        }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || '업로드에 실패했습니다.')
      }

      alert('업로드가 완료되었습니다!')
      router.push('/my-videos')
    } catch (err: any) {
      setError(err.message)
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <main className="container mx-auto px-4 py-16">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8">영상 업로드</h1>
        <form onSubmit={handleUpload} className="space-y-6">
          <div>
            <label
              htmlFor="title"
              className="block text-sm font-medium text-slate-700"
            >
              영상 제목
            </label>
            <input
              type="text"
              id="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="mt-1 block w-full px-3 py-2 bg-white border border-slate-300 rounded-md shadow-sm placeholder-slate-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              placeholder="영상 제목을 입력하세요"
              required
            />
          </div>

          <div>
            <label
              htmlFor="video"
              className="block text-sm font-medium text-slate-700"
            >
              비디오 파일
            </label>
            <input
              type="file"
              id="video"
              accept="video/*"
              onChange={handleFileChange}
              className="mt-1 block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              required
            />
          </div>

          {file && (
            <div>
              <video
                ref={videoRef}
                onLoadedMetadata={captureThumbnails}
                controls
                muted
                className="w-full rounded-lg bg-slate-900"
              ></video>
              <canvas ref={canvasRef} className="hidden"></canvas>
            </div>
          )}

          {thumbnails.length > 0 && (
            <div>
              <h3 className="text-lg font-medium text-slate-800 mb-2">
                썸네일 선택
              </h3>
              <div className="grid grid-cols-3 sm:grid-cols-5 gap-4">
                {thumbnails.map((thumb, index) => (
                  <img
                    key={index}
                    src={thumb}
                    alt={`Thumbnail ${index + 1}`}
                    onClick={() => setSelectedThumbnail(thumb)}
                    className={`cursor-pointer rounded-md transition-all duration-200 ${
                      selectedThumbnail === thumb
                        ? 'ring-4 ring-blue-500 ring-offset-2'
                        : 'hover:scale-105'
                    }`}
                  />
                ))}
              </div>
            </div>
          )}

          {error && (
            <p className="text-red-500 text-sm text-center mt-4">{error}</p>
          )}

          <button
            type="submit"
            disabled={isProcessing}
            className="w-full inline-flex items-center justify-center rounded-lg bg-blue-600 px-8 py-3 text-base font-medium text-white shadow-lg hover:bg-blue-700 transition-colors disabled:bg-slate-400 disabled:cursor-not-allowed"
          >
            {isProcessing ? '업로드 중...' : '업로드하기'}
          </button>
        </form>
      </div>
    </main>
  )
}
