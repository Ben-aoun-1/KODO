import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ChevronRight,
  ChevronDown,
  File,
  Folder,
  FolderOpen,
  Search,
  Code,
  Box,
  FileCode,
} from 'lucide-react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { useAppStore } from '@/store'
import api from '@/api/client'
import type { FileNode, CodeEntity, SearchResult } from '@/types'
import { cn } from '@/lib/utils'

interface TreeNodeProps {
  node: FileNode
  level: number
  onSelect: (node: FileNode) => void
  selectedPath: string | null
}

function TreeNode({ node, level, onSelect, selectedPath }: TreeNodeProps) {
  const [isOpen, setIsOpen] = useState(level < 2)
  const isDirectory = node.type === 'directory'
  const isSelected = selectedPath === node.path

  const handleClick = () => {
    if (isDirectory) {
      setIsOpen(!isOpen)
    } else {
      onSelect(node)
    }
  }

  return (
    <div>
      <button
        onClick={handleClick}
        className={cn(
          'flex items-center gap-1 w-full px-2 py-1 text-sm hover:bg-accent rounded-sm transition-colors',
          isSelected && 'bg-accent'
        )}
        style={{ paddingLeft: `${level * 12 + 8}px` }}
      >
        {isDirectory ? (
          <>
            {isOpen ? (
              <ChevronDown className="h-4 w-4 shrink-0" />
            ) : (
              <ChevronRight className="h-4 w-4 shrink-0" />
            )}
            {isOpen ? (
              <FolderOpen className="h-4 w-4 shrink-0 text-yellow-600" />
            ) : (
              <Folder className="h-4 w-4 shrink-0 text-yellow-600" />
            )}
          </>
        ) : (
          <>
            <span className="w-4" />
            <File className="h-4 w-4 shrink-0 text-muted-foreground" />
          </>
        )}
        <span className="truncate">{node.name}</span>
        {node.entityCount && node.entityCount > 0 && (
          <span className="ml-auto text-xs text-muted-foreground">
            {node.entityCount}
          </span>
        )}
      </button>
      {isDirectory && isOpen && node.children && (
        <div>
          {node.children.map((child) => (
            <TreeNode
              key={child.path}
              node={child}
              level={level + 1}
              onSelect={onSelect}
              selectedPath={selectedPath}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export function Explorer() {
  const navigate = useNavigate()
  const { currentRepo, theme, selectEntity } = useAppStore()
  const [fileTree, setFileTree] = useState<FileNode | null>(null)
  const [selectedFile, setSelectedFile] = useState<FileNode | null>(null)
  const [entities, setEntities] = useState<CodeEntity[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [selectedEntity, setSelectedEntityLocal] = useState<CodeEntity | null>(null)

  useEffect(() => {
    if (currentRepo) {
      loadFileTree()
    }
  }, [currentRepo])

  const loadFileTree = async () => {
    if (!currentRepo) return
    try {
      const tree = await api.getFileTree(currentRepo.id)
      setFileTree(tree)
    } catch (error) {
      console.error('Failed to load file tree:', error)
    }
  }

  const handleFileSelect = async (node: FileNode) => {
    if (node.type !== 'file' || !currentRepo) return
    setSelectedFile(node)
    setSelectedEntityLocal(null)

    try {
      const fileEntities = await api.getEntitiesInFile(currentRepo.id, node.path)
      setEntities(fileEntities)
    } catch (error) {
      console.error('Failed to load entities:', error)
      setEntities([])
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim() || !currentRepo) return

    setIsSearching(true)
    try {
      const results = await api.search(currentRepo.id, searchQuery, { limit: 20 })
      setSearchResults(results)
    } catch (error) {
      console.error('Search failed:', error)
      setSearchResults([])
    } finally {
      setIsSearching(false)
    }
  }

  const handleEntityClick = (entity: CodeEntity) => {
    setSelectedEntityLocal(entity)
    selectEntity(entity)
  }

  const handleViewInGraph = (entity: CodeEntity) => {
    selectEntity(entity)
    navigate('/graph')
  }

  const getEntityIcon = (type: string) => {
    switch (type) {
      case 'function':
      case 'method':
        return <Code className="h-4 w-4 text-blue-500" />
      case 'class':
        return <Box className="h-4 w-4 text-yellow-500" />
      default:
        return <FileCode className="h-4 w-4 text-gray-500" />
    }
  }

  if (!currentRepo) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <p className="text-muted-foreground">Select a repository first</p>
        <Button onClick={() => navigate('/repositories')}>
          Go to Repositories
        </Button>
      </div>
    )
  }

  return (
    <div className="h-[calc(100vh-8rem)]">
      <div className="flex items-center gap-4 mb-4">
        <h1 className="text-3xl font-bold tracking-tight">Code Explorer</h1>
        <div className="flex-1 flex gap-2 max-w-md">
          <Input
            placeholder="Search code..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          />
          <Button onClick={handleSearch} disabled={isSearching}>
            <Search className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4 h-[calc(100%-3rem)]">
        {/* File Tree */}
        <Card className="col-span-3 overflow-hidden">
          <CardHeader className="py-3">
            <CardTitle className="text-sm">Files</CardTitle>
          </CardHeader>
          <ScrollArea className="h-[calc(100%-3rem)]">
            <CardContent className="p-0">
              {fileTree ? (
                <TreeNode
                  node={fileTree}
                  level={0}
                  onSelect={handleFileSelect}
                  selectedPath={selectedFile?.path || null}
                />
              ) : (
                <p className="text-sm text-muted-foreground p-4">Loading...</p>
              )}
            </CardContent>
          </ScrollArea>
        </Card>

        {/* Code View / Search Results */}
        <Card className="col-span-6 overflow-hidden">
          <CardHeader className="py-3">
            <CardTitle className="text-sm">
              {searchResults.length > 0
                ? `Search Results (${searchResults.length})`
                : selectedFile?.path || 'Select a file'}
            </CardTitle>
          </CardHeader>
          <ScrollArea className="h-[calc(100%-3rem)]">
            <CardContent className="p-0">
              {searchResults.length > 0 ? (
                <div className="divide-y">
                  {searchResults.map((result) => (
                    <button
                      key={result.entity.id}
                      className="w-full p-4 text-left hover:bg-accent transition-colors"
                      onClick={() => handleEntityClick(result.entity)}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        {getEntityIcon(result.entity.type)}
                        <span className="font-medium">{result.entity.name}</span>
                        <span className="text-xs text-muted-foreground">
                          {result.entity.type}
                        </span>
                        <span className="ml-auto text-xs text-muted-foreground">
                          Score: {result.score.toFixed(2)}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground truncate">
                        {result.entity.filePath}:{result.entity.startLine}
                      </p>
                    </button>
                  ))}
                </div>
              ) : selectedEntity ? (
                <SyntaxHighlighter
                  language={selectedEntity.language}
                  style={theme === 'dark' ? oneDark : oneLight}
                  showLineNumbers
                  startingLineNumber={selectedEntity.startLine}
                  customStyle={{ margin: 0, fontSize: '0.875rem' }}
                >
                  {selectedEntity.sourceCode}
                </SyntaxHighlighter>
              ) : selectedFile ? (
                <div className="p-4 text-sm text-muted-foreground">
                  Select an entity from the sidebar to view its code
                </div>
              ) : (
                <div className="p-4 text-sm text-muted-foreground">
                  Select a file from the tree to explore
                </div>
              )}
            </CardContent>
          </ScrollArea>
        </Card>

        {/* Entity List */}
        <Card className="col-span-3 overflow-hidden">
          <CardHeader className="py-3">
            <CardTitle className="text-sm">
              Entities {entities.length > 0 && `(${entities.length})`}
            </CardTitle>
          </CardHeader>
          <ScrollArea className="h-[calc(100%-3rem)]">
            <CardContent className="p-0">
              {entities.length > 0 ? (
                <div className="divide-y">
                  {entities.map((entity) => (
                    <div key={entity.id} className="p-3">
                      <button
                        className={cn(
                          'w-full text-left rounded-md p-2 transition-colors',
                          selectedEntity?.id === entity.id
                            ? 'bg-accent'
                            : 'hover:bg-accent/50'
                        )}
                        onClick={() => handleEntityClick(entity)}
                      >
                        <div className="flex items-center gap-2">
                          {getEntityIcon(entity.type)}
                          <span className="font-medium text-sm truncate">
                            {entity.name}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          Lines {entity.startLine}-{entity.endLine}
                        </p>
                      </button>
                      <div className="flex gap-1 mt-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-xs h-7"
                          onClick={() => handleViewInGraph(entity)}
                        >
                          Call Graph
                        </Button>
                      </div>
                      {entity.docstring && (
                        <>
                          <Separator className="my-2" />
                          <p className="text-xs text-muted-foreground line-clamp-2">
                            {entity.docstring}
                          </p>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              ) : selectedFile ? (
                <p className="text-sm text-muted-foreground p-4">
                  No entities found in this file
                </p>
              ) : (
                <p className="text-sm text-muted-foreground p-4">
                  Select a file to see its entities
                </p>
              )}
            </CardContent>
          </ScrollArea>
        </Card>
      </div>
    </div>
  )
}
