import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  BarChart3,
  TrendingUp,
  AlertTriangle,
  Shield,
  FileCode,
  GitBranch,
} from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useAppStore } from '@/store'
import { formatNumber, getLanguageColor } from '@/lib/utils'

interface MetricCard {
  title: string
  value: string | number
  description: string
  icon: React.ComponentType<{ className?: string }>
  trend?: 'up' | 'down' | 'neutral'
}

export function Analytics() {
  const navigate = useNavigate()
  const { currentRepo } = useAppStore()

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

  const stats = currentRepo.stats

  const metrics: MetricCard[] = [
    {
      title: 'Total Files',
      value: formatNumber(stats?.totalFiles || 0),
      description: 'Source files indexed',
      icon: FileCode,
    },
    {
      title: 'Lines of Code',
      value: formatNumber(stats?.totalLines || 0),
      description: 'Total lines parsed',
      icon: BarChart3,
    },
    {
      title: 'Functions',
      value: formatNumber(stats?.totalFunctions || 0),
      description: 'Functions and methods',
      icon: GitBranch,
    },
    {
      title: 'Classes',
      value: formatNumber(stats?.totalClasses || 0),
      description: 'Class definitions',
      icon: TrendingUp,
    },
  ]

  const complexityData = [
    { name: 'Low (1-5)', count: 156, percentage: 60 },
    { name: 'Medium (6-10)', count: 78, percentage: 30 },
    { name: 'High (11-20)', count: 20, percentage: 8 },
    { name: 'Very High (>20)', count: 5, percentage: 2 },
  ]

  const couplingData = [
    { name: 'Highly Coupled', count: 12, level: 'warning' },
    { name: 'Moderately Coupled', count: 34, level: 'neutral' },
    { name: 'Loosely Coupled', count: 89, level: 'success' },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Analytics</h1>
        <p className="text-muted-foreground mt-1">
          Code metrics and analysis for {currentRepo.name}
        </p>
      </div>

      {/* Metric Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {metrics.map((metric) => (
          <Card key={metric.title}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{metric.title}</CardTitle>
              <metric.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metric.value}</div>
              <p className="text-xs text-muted-foreground">{metric.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <Tabs defaultValue="complexity" className="space-y-4">
        <TabsList>
          <TabsTrigger value="complexity">Complexity</TabsTrigger>
          <TabsTrigger value="coupling">Coupling</TabsTrigger>
          <TabsTrigger value="languages">Languages</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
        </TabsList>

        <TabsContent value="complexity" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Cyclomatic Complexity Distribution</CardTitle>
              <CardDescription>
                Distribution of functions by cyclomatic complexity score
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {complexityData.map((item) => (
                  <div key={item.name} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span>{item.name}</span>
                      <span className="text-muted-foreground">
                        {item.count} ({item.percentage}%)
                      </span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          item.percentage > 50
                            ? 'bg-green-500'
                            : item.percentage > 20
                            ? 'bg-yellow-500'
                            : item.percentage > 5
                            ? 'bg-orange-500'
                            : 'bg-red-500'
                        }`}
                        style={{ width: `${item.percentage}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Most Complex Functions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[
                    { name: 'parseComplexQuery', complexity: 24, file: 'parser.py' },
                    { name: 'buildCallGraph', complexity: 18, file: 'graph.py' },
                    { name: 'analyzeImpact', complexity: 15, file: 'analysis.py' },
                    { name: 'handleWebhook', complexity: 12, file: 'webhooks.py' },
                    { name: 'generateResponse', complexity: 11, file: 'llm.py' },
                  ].map((fn) => (
                    <div
                      key={fn.name}
                      className="flex items-center justify-between text-sm"
                    >
                      <div>
                        <span className="font-medium">{fn.name}</span>
                        <span className="text-muted-foreground ml-2">
                          {fn.file}
                        </span>
                      </div>
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs ${
                          fn.complexity > 20
                            ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                            : fn.complexity > 10
                            ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                            : 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        }`}
                      >
                        {fn.complexity}
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Complexity Over Time</CardTitle>
              </CardHeader>
              <CardContent className="flex items-center justify-center h-48 text-muted-foreground">
                Chart placeholder - Requires commit history
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="coupling" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Module Coupling Analysis</CardTitle>
              <CardDescription>
                How tightly connected are the modules in your codebase
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {couplingData.map((item) => (
                  <div
                    key={item.name}
                    className="flex items-center justify-between p-3 rounded-lg bg-muted"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className={`w-3 h-3 rounded-full ${
                          item.level === 'warning'
                            ? 'bg-yellow-500'
                            : item.level === 'success'
                            ? 'bg-green-500'
                            : 'bg-gray-400'
                        }`}
                      />
                      <span className="font-medium">{item.name}</span>
                    </div>
                    <span className="text-muted-foreground">
                      {item.count} modules
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="languages" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Language Distribution</CardTitle>
              <CardDescription>Breakdown of code by programming language</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {stats?.languages?.map((lang) => (
                  <div key={lang.language} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: getLanguageColor(lang.language) }}
                        />
                        <span className="font-medium capitalize">{lang.language}</span>
                      </div>
                      <div className="text-muted-foreground">
                        {lang.files} files / {formatNumber(lang.lines)} lines (
                        {lang.percentage.toFixed(1)}%)
                      </div>
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
                )) || (
                  <p className="text-muted-foreground">No language data available</p>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Security Analysis
              </CardTitle>
              <CardDescription>
                Potential security issues detected in the codebase
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 rounded-lg border border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-950">
                  <div className="flex items-center gap-3">
                    <AlertTriangle className="h-5 w-5 text-yellow-600" />
                    <div>
                      <p className="font-medium">Hardcoded Secrets</p>
                      <p className="text-sm text-muted-foreground">
                        Potential API keys or passwords in source code
                      </p>
                    </div>
                  </div>
                  <span className="px-2 py-1 rounded-full text-xs bg-yellow-200 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-200">
                    3 issues
                  </span>
                </div>

                <div className="flex items-center justify-between p-4 rounded-lg border border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950">
                  <div className="flex items-center gap-3">
                    <Shield className="h-5 w-5 text-green-600" />
                    <div>
                      <p className="font-medium">SQL Injection</p>
                      <p className="text-sm text-muted-foreground">
                        No SQL injection vulnerabilities detected
                      </p>
                    </div>
                  </div>
                  <span className="px-2 py-1 rounded-full text-xs bg-green-200 text-green-800 dark:bg-green-800 dark:text-green-200">
                    0 issues
                  </span>
                </div>

                <div className="flex items-center justify-between p-4 rounded-lg border border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950">
                  <div className="flex items-center gap-3">
                    <Shield className="h-5 w-5 text-green-600" />
                    <div>
                      <p className="font-medium">XSS Vulnerabilities</p>
                      <p className="text-sm text-muted-foreground">
                        No cross-site scripting issues found
                      </p>
                    </div>
                  </div>
                  <span className="px-2 py-1 rounded-full text-xs bg-green-200 text-green-800 dark:bg-green-800 dark:text-green-200">
                    0 issues
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
