import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Pochi Control',
  description: 'Pochi 12-joint live hardware viewer and MIT controller',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
