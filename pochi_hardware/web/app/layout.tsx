import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Pochi Telemetry',
  description: 'Read-only Pochi encoder and IMU posture viewer',
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
