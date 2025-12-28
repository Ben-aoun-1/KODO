import { useEffect, useState } from 'react'
import { Plus, RefreshCw, Trash2, ExternalLink, Check } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { useAppStore } from '@/store'
import { formatDate, getLanguageColor } from '@/lib/utils'
import api from '@/api/client'
import type { Repository } from '@/types'

export function Repositories() {
  const {
    repositories,
    currentRepo,
    isLoadingRepos,
    repoError,
    fetchRepositories,
    selectRepository,
    indexRepository,
  } = useAppStore()

  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [newRepoUrl, setNewRepoUrl] = useState('')
  const [isIndexing, setIsIndexing] = useState(false)

  useEffect(() => {
    fetchRepositories()
  }, [fetchRepositories])

  const handleAddRepository = async () => {
    if (!newRepoUrl.trim()) return

    setIsIndexing(true)
    try {
      await indexRepository(newRepoUrl)
      setNewRepoUrl('')
      setIsDialogOpen(false)
    } finally {
      setIsIndexing(false)
    }
  }

  const handleRefresh = async (repo: Repository) => {
    try {
      await api.refreshRepository(repo.id)
      fetchRepositories()
    } catch (error) {
      console.error('Failed to refresh repository:', error)
    }
  }

  const handleDelete = async (repo: Repository) => {
    if (!confirm(`Are you sure you want to delete ${repo.name}?`)) return

    try {
      await api.deleteRepository(repo.id)
      fetchRepositories()
    } catch (error) {
      console.error('Failed to delete repository:', error)
    }
  }

  const getStatusBadge = (status: Repository['status']) => {
    const styles = {
      ready: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
      indexing: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
      pending: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200',
      error: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    }

    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${styles[status]}`}>
        {status}
      </span>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Repositories</h1>
          <p className="text-muted-foreground mt-1">
            Manage your indexed repositories
          </p>
        </div>

        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              Add Repository
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add Repository</DialogTitle>
              <DialogDescription>
                Enter a GitHub repository URL to index and analyze.
              </DialogDescription>
            </DialogHeader>
            <div className="py-4">
              <Input
                placeholder="https://github.com/owner/repo"
                value={newRepoUrl}
                onChange={(e) => setNewRepoUrl(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleAddRepository()}
              />
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleAddRepository} disabled={isIndexing}>
                {isIndexing ? (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                    Indexing...
                  </>
                ) : (
                  'Add Repository'
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {repoError && (
        <div className="bg-destructive/10 text-destructive px-4 py-3 rounded-lg">
          {repoError}
        </div>
      )}

      {isLoadingRepos ? (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : repositories.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <p className="text-muted-foreground mb-4">
              No repositories indexed yet
            </p>
            <Button onClick={() => setIsDialogOpen(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Add Your First Repository
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4">
          {repositories.map((repo) => (
            <Card
              key={repo.id}
              className={`transition-colors ${
                currentRepo?.id === repo.id
                  ? 'border-primary'
                  : 'hover:border-primary/50'
              }`}
            >
              <CardHeader className="flex flex-row items-start justify-between space-y-0">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <CardTitle className="text-lg">{repo.name}</CardTitle>
                    {getStatusBadge(repo.status)}
                    {currentRepo?.id === repo.id && (
                      <span className="flex items-center gap-1 text-xs text-primary">
                        <Check className="h-3 w-3" />
                        Selected
                      </span>
                    )}
                  </div>
                  <CardDescription className="flex items-center gap-2">
                    <a
                      href={repo.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:underline flex items-center gap-1"
                    >
                      {repo.url}
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => handleRefresh(repo)}
                    disabled={repo.status === 'indexing'}
                  >
                    <RefreshCw
                      className={`h-4 w-4 ${
                        repo.status === 'indexing' ? 'animate-spin' : ''
                      }`}
                    />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => handleDelete(repo)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div className="flex gap-4 text-sm text-muted-foreground">
                    <span>{repo.stats?.totalFiles || 0} files</span>
                    <span>{repo.stats?.totalFunctions || 0} functions</span>
                    <span>{repo.stats?.totalClasses || 0} classes</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {repo.stats?.languages?.slice(0, 3).map((lang) => (
                      <div
                        key={lang.language}
                        className="flex items-center gap-1 text-xs"
                      >
                        <div
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: getLanguageColor(lang.language) }}
                        />
                        <span className="capitalize">{lang.language}</span>
                      </div>
                    ))}
                  </div>
                </div>
                {repo.lastIndexed && (
                  <p className="text-xs text-muted-foreground mt-2">
                    Last indexed: {formatDate(repo.lastIndexed)}
                  </p>
                )}
                <div className="mt-4">
                  <Button
                    variant={currentRepo?.id === repo.id ? 'secondary' : 'default'}
                    size="sm"
                    onClick={() => selectRepository(repo)}
                    disabled={repo.status !== 'ready'}
                  >
                    {currentRepo?.id === repo.id ? 'Currently Selected' : 'Select Repository'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
