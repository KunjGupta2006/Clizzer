import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Clizzer - AI Multimodal Video Clipper",
  description: "Extract top viral clips with GTA 6 context, facial tracking, and smart captions",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-[#0b0c10] text-[#c5c6c7] antialiased min-h-screen">
        <header className="border-b border-gray-800 bg-[#121318]/80 backdrop-blur sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="text-2xl font-black bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-400 bg-clip-text text-transparent">
              CLIZZER
            </span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-pink-500/10 text-pink-400 border border-pink-500/20 font-mono">
              AI VIRAL CLIPPER
            </span>
          </div>
          <div className="text-xs text-gray-500 font-mono">
            Multimodal AI Engine
          </div>
        </header>
        <main className="max-w-7xl mx-auto px-4 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
