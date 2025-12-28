import { NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  FolderTree,
  MessageSquare,
  GitBranch,
  BarChart3,
  Database,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'

const navItems = [
  {
    title: 'Dashboard',
    href: '/',
    icon: LayoutDashboard,
  },
  {
    title: 'Explorer',
    href: '/explorer',
    icon: FolderTree,
  },
  {
    title: 'Chat',
    href: '/chat',
    icon: MessageSquare,
  },
  {
    title: 'Call Graph',
    href: '/graph',
    icon: GitBranch,
  },
  {
    title: 'Analytics',
    href: '/analytics',
    icon: BarChart3,
  },
  {
    title: 'Repositories',
    href: '/repositories',
    icon: Database,
  },
]

export function Sidebar() {
  const location = useLocation()

  return (
    <aside className="hidden md:flex w-64 flex-col border-r bg-background">
      <ScrollArea className="flex-1 py-4">
        <nav className="grid gap-1 px-2">
          {navItems.map((item) => {
            const isActive = location.pathname === item.href
            return (
              <NavLink
                key={item.href}
                to={item.href}
                className={cn(
                  'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-secondary text-secondary-foreground'
                    : 'text-muted-foreground hover:bg-secondary/50 hover:text-secondary-foreground'
                )}
              >
                <item.icon className="h-4 w-4" />
                {item.title}
              </NavLink>
            )
          })}
        </nav>

        <Separator className="my-4" />

        <div className="px-4">
          <h4 className="mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Quick Stats
          </h4>
          <div className="grid gap-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Files</span>
              <span className="font-medium">--</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Functions</span>
              <span className="font-medium">--</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Classes</span>
              <span className="font-medium">--</span>
            </div>
          </div>
        </div>
      </ScrollArea>

      <div className="border-t p-4">
        <div className="text-xs text-muted-foreground">
          <p>Kodo v0.1.0</p>
          <p className="mt-1">Code-Aware AI Assistant</p>
        </div>
      </div>
    </aside>
  )
}
