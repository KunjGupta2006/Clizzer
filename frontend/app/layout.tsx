import type { Metadata } from 'next';
import { Big_Shoulders_Display, Black_Ops_One } from 'next/font/google';
import './globals.css';

const bigShoulders = Big_Shoulders_Display({
  weight: ['700', '900'],
  subsets: ['latin'],
  variable: '--font-big-shoulders',
  display: 'swap',
});

const blackOpsOne = Black_Ops_One({
  weight: '400',
  subsets: ['latin'],
  variable: '--font-black-ops',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'CLIZZER — GTA Video Clipper',
  description: 'Paste a link. Get fire clips. Zero subscriptions.',
  keywords: ['video clipper', 'GTA Vice City', 'viral clips', 'AI clipper', 'shorts generator'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${bigShoulders.variable} ${blackOpsOne.variable} dark`}
    >
      <body className="bg-[#0b0b0b] text-white min-h-screen font-gta selection:bg-[#ff007f] selection:text-white antialiased overflow-x-hidden">
        {/* Subtle noise and scanline effects applied globally */}
        <div className="fixed inset-0 pointer-events-none noise-overlay z-40" />
        <div className="relative z-10 flex min-h-screen flex-col">
          {children}
        </div>
      </body>
    </html>
  );
}
