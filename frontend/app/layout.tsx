import './globals.css';
import Link from 'next/link';
import { LayoutDashboard, Code2, Terminal, Activity, Settings, Cpu } from 'lucide-react';

export const metadata = {
  title: 'OptionSim - 专业期权量化平台',
  description: 'AI-Driven Options Quantitative Research Platform',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="bg-background text-foreground flex h-screen overflow-hidden" suppressHydrationWarning>
        {/* Sidebar Navigation */}
        <aside className="w-16 flex flex-col items-center bg-card-bg border-r border-card-border py-4 z-50">
          <div className="mb-8 p-2 bg-primary/20 rounded-xl">
            <Cpu className="text-primary w-6 h-6" />
          </div>
          
          <nav className="flex flex-col gap-4 flex-1">
            <Link href="/" className="p-3 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all tooltip-trigger" title="Foundry">
              <Code2 className="w-5 h-5" />
            </Link>
            <Link href="/cockpit" className="p-3 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all tooltip-trigger" title="Chronos Cockpit">
               <Activity className="w-5 h-5" />
            </Link>
          </nav>
          
          <div className="mt-auto">
             <button className="p-3 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white">
                <Settings className="w-5 h-5" />
             </button>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 relative overflow-hidden">
          {children}
        </main>
      </body>
    </html>
  )
}
