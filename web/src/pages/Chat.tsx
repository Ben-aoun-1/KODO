import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Send, Trash2, Code, FileCode, ExternalLink } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { useAppStore } from '@/store'
import { cn } from '@/lib/utils'
import type { ChatMessage, CodeSource } from '@/types'

function SourceCard({ source }: { source: CodeSource }) {
  return (
    <div className="border rounded-lg p-3 hover:bg-accent/50 transition-colors">
      <div className="flex items-center gap-2 mb-1">
        <Code className="h-4 w-4 text-blue-500" />
        <span className="font-medium text-sm">{source.entityName}</span>
        <span className="text-xs text-muted-foreground px-1.5 py-0.5 bg-muted rounded">
          {source.entityType}
        </span>
      </div>
      <p className="text-xs text-muted-foreground flex items-center gap-1">
        <FileCode className="h-3 w-3" />
        {source.filePath}:{source.startLine}
      </p>
      {source.snippet && (
        <pre className="mt-2 text-xs bg-muted p-2 rounded overflow-x-auto">
          <code>{source.snippet}</code>
        </pre>
      )}
    </div>
  )
}

function MessageBubble({
  message,
  theme,
}: {
  message: ChatMessage
  theme: 'light' | 'dark'
}) {
  const isUser = message.role === 'user'

  return (
    <div className={cn('flex gap-3', isUser && 'flex-row-reverse')}>
      <Avatar className="h-8 w-8 shrink-0">
        <AvatarFallback className={isUser ? 'bg-primary text-primary-foreground' : 'bg-secondary'}>
          {isUser ? 'U' : 'K'}
        </AvatarFallback>
      </Avatar>
      <div
        className={cn(
          'flex-1 max-w-[80%] rounded-lg p-4',
          isUser ? 'bg-primary text-primary-foreground' : 'bg-muted'
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <ReactMarkdown
              components={{
                code({ className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '')
                  const isInline = !match
                  return isInline ? (
                    <code className="bg-background/50 px-1 py-0.5 rounded text-sm" {...props}>
                      {children}
                    </code>
                  ) : (
                    <SyntaxHighlighter
                      style={theme === 'dark' ? oneDark : oneLight}
                      language={match[1]}
                      PreTag="div"
                      customStyle={{ fontSize: '0.875rem', margin: '0.5rem 0' }}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  )
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {message.sources && message.sources.length > 0 && (
          <>
            <Separator className="my-3" />
            <div className="space-y-2">
              <p className="text-xs font-medium text-muted-foreground">
                Sources ({message.sources.length})
              </p>
              <div className="grid gap-2">
                {message.sources.slice(0, 3).map((source) => (
                  <SourceCard key={source.entityId} source={source} />
                ))}
              </div>
            </div>
          </>
        )}

        <p className="text-xs text-muted-foreground mt-2">
          {message.timestamp.toLocaleTimeString()}
        </p>
      </div>
    </div>
  )
}

const suggestedQuestions = [
  'What does this codebase do?',
  'Where is the main entry point?',
  'How is authentication handled?',
  'What are the main data models?',
  'Show me the API endpoints',
]

export function Chat() {
  const navigate = useNavigate()
  const { currentRepo, messages, isQuerying, theme, sendMessage, clearMessages } =
    useAppStore()
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || isQuerying) return
    const message = input
    setInput('')
    await sendMessage(message)
  }

  const handleSuggestedQuestion = (question: string) => {
    setInput(question)
    inputRef.current?.focus()
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
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Chat</h1>
          <p className="text-muted-foreground mt-1">
            Ask questions about {currentRepo.name}
          </p>
        </div>
        {messages.length > 0 && (
          <Button variant="outline" size="sm" onClick={clearMessages}>
            <Trash2 className="h-4 w-4 mr-2" />
            Clear Chat
          </Button>
        )}
      </div>

      <Card className="flex-1 flex flex-col overflow-hidden">
        <ScrollArea className="flex-1 p-4" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-6">
              <div className="text-center space-y-2">
                <h2 className="text-xl font-semibold">
                  Ask anything about your code
                </h2>
                <p className="text-muted-foreground">
                  Kodo understands your codebase and can answer questions in natural language
                </p>
              </div>
              <div className="flex flex-wrap justify-center gap-2 max-w-2xl">
                {suggestedQuestions.map((question) => (
                  <Button
                    key={question}
                    variant="outline"
                    size="sm"
                    onClick={() => handleSuggestedQuestion(question)}
                  >
                    {question}
                  </Button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} theme={theme} />
              ))}
              {isQuerying && (
                <div className="flex gap-3">
                  <Avatar className="h-8 w-8">
                    <AvatarFallback className="bg-secondary">K</AvatarFallback>
                  </Avatar>
                  <div className="bg-muted rounded-lg p-4">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:0.2s]" />
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:0.4s]" />
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </ScrollArea>

        <CardContent className="border-t p-4">
          <form
            onSubmit={(e) => {
              e.preventDefault()
              handleSend()
            }}
            className="flex gap-2"
          >
            <Input
              ref={inputRef}
              placeholder="Ask a question about your code..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isQuerying}
              className="flex-1"
            />
            <Button type="submit" disabled={isQuerying || !input.trim()}>
              <Send className="h-4 w-4" />
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
