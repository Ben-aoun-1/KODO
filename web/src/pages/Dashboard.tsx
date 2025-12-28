import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  FolderTree,
  FileCode,
  Box,
  GitBranch,
  MessageSquare,
  TrendingUp,
  Clock,
  AlertCircle,
} from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/store'
import { formatDate, formatNumber, getLanguageColor } from '@/lib/utils'

export function Dashboard() {
  const { currentRepo, repositories, fetchRepositories, isLoadingRepos } = useAppStore()

  useEffect(() => {
    if (repositories.length === 0) {
      fetchRepositories()
    }
  }, [repositories.length, fetchRepositories])

  if (!currentRepo) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <div className="text-center space-y-2">
          <h1 className="text-2xl font-bold">Welcome to Kodo</h1>
          <p className="text-muted-foreground">
            Select a repository to get started
          </p>
        </div>
        <Button asChild>
          <Link to="/repositories">View Repositories</Link>
        </Button>
      </div>
    )
  }

  const stats = currentRepo.stats

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">{currentRepo.name}</h1>
          <p className="text-muted-foreground mt-1">
            {currentRepo.url}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {currentRepo.status === 'indexing' && (
            <span className="flex items-center gap-2 text-sm text-yellow-600 dark:text-yellow-400">
              <Clock className="h-4 w-4 animate-spin" />
              Indexing...
            </span>
          )}
          {currentRepo.status === 'error' && (
            <span className="flex items-center gap-2 text-sm text-destructive">
              <AlertCircle className="h-4 w-4" />
              Error
            </span>
          )}
          {currentRepo.lastIndexed && (
            <span className="text-sm text-muted-foreground">
              Last indexed: {formatDate(currentRepo.lastIndexed)}
            </span>
          )}
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Files</CardTitle>
            <FolderTree className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatNumber(stats?.totalFiles || 0)}
            </div>
            <p className="text-xs text-muted-foreground">
              Across all directories
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Lines of Code</CardTitle>
            <FileCode className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatNumber(stats?.totalLines || 0)}
            </div>
            <p className="text-xs text-muted-foreground">
              Total source code
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Functions</CardTitle>
            <Box className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatNumber(stats?.totalFunctions || 0)}
            </div>
            <p className="text-xs text-muted-foreground">
              Parsed and indexed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Classes</CardTitle>
            <GitBranch className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatNumber(stats?.totalClasses || 0)}
            </div>
            <p className="text-xs text-muted-foreground">
              With inheritance tracked
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Languages and Quick Actions */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Language Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Languages</CardTitle>
            <CardDescription>Distribution of source code by language</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {stats?.languages?.length > 0 ? (
                stats.languages.map((lang) => (
                  <div key={lang.language} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: getLanguageColor(lang.language) }}
                        />
                        <span className="font-medium capitalize">{lang.language}</span>
                      </div>
                      <span className="text-muted-foreground">
                        {lang.percentage.toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${lang.percentage}%`,
                          backgroundColor: getLanguageColor(lang.language),
                        }}
                      />
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-sm text-muted-foreground">No language data available</p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
            <CardDescription>Common tasks for this repository</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3">
            <Button variant="outline" className="justify-start" asChild>
              <Link to="/chat">
                <MessageSquare className="mr-2 h-4 w-4" />
                Ask a Question
              </Link>
            </Button>
            <Button variant="outline" className="justify-start" asChild>
              <Link to="/explorer">
                <FolderTree className="mr-2 h-4 w-4" />
                Browse Code
              </Link>
            </Button>
            <Button variant="outline" className="justify-start" asChild>
              <Link to="/graph">
                <GitBranch className="mr-2 h-4 w-4" />
                View Call Graph
              </Link>
            </Button>
            <Button variant="outline" className="justify-start" asChild>
              <Link to="/analytics">
                <TrendingUp className="mr-2 h-4 w-4" />
                View Analytics
              </Link>
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity placeholder */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Queries</CardTitle>
          <CardDescription>Your recent questions about this codebase</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No recent queries. Start by asking a question about your code!
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
